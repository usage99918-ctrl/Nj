"""Make a frozen (cloudpickled) live model safe to run on any Python version.

Why this exists
---------------
cloudpickle stores functions that cannot be imported by name -- the rule lambdas built in
`library.rules(h)` and the `make` factories of model strategies -- *by value*, i.e. as raw
Python bytecode. Bytecode is specific to the Python minor version. The live models were
frozen under Python 3.13; executing that bytecode under Python 3.12 (the Windows PC) produced
garbage instead of a clean error:

    B_60min / G2_60min  (rule rsi_rev_60)  -> ValueError: all keys need to be the same shape
    G2_240min           (rule rv_rev_10)   -> SystemError: no locals found when setting up annotations
    G2_1440min          (rule quiet_drift) -> SystemError: no locals when deleting 'np'

A rule strategy is fully defined by its name in the versioned strategy library, so the score
function is rebuilt from the library instead of executing stored bytecode. The model FILE is
not changed (its SHA-256 still matches the ledger), and the library definition is checked
against the frozen one (default arguments, referenced names, string constants; on the same
Python version also the exact bytecode). Stored functions that are not needed for live
prediction (e.g. the `make` factory, only used when fitting) are replaced by a guard that
raises if ever called, so no stored bytecode is executed at all.
"""
from __future__ import annotations

import sys
import types

from cryptolab.strategies.library import Strategy, rules


class FrozenModelError(RuntimeError):
    pass


def _is_by_value(fn) -> bool:
    """True for functions cloudpickle had to store as bytecode (lambdas / nested functions)."""
    return isinstance(fn, types.FunctionType) and (fn.__name__ == "<lambda>" or "<locals>" in fn.__qualname__)


def _str_consts(code) -> tuple:
    out = []
    for c in code.co_consts:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, types.CodeType):
            out.extend(_str_consts(c))
    return tuple(out)


def _check_same_rule(name: str, frozen, lib) -> dict:
    """Version-independent comparison of a frozen rule function with the library one."""
    fc, lc = frozen.__code__, lib.__code__
    checks = {
        "defaults": frozen.__defaults__ == lib.__defaults__,
        "names": tuple(fc.co_names) == tuple(lc.co_names),
        "string_constants": _str_consts(fc) == _str_consts(lc),
        "arg_count": fc.co_argcount == lc.co_argcount,
    }
    if not all(checks.values()):
        bad = [k for k, ok in checks.items() if not ok]
        raise FrozenModelError(f"library rule '{name}' differs from the frozen model ({', '.join(bad)}); "
                               "refusing to substitute it")
    # Same Python version as the one that froze the model -> bytecode must be identical too.
    checks["bytecode_identical"] = fc.co_code == lc.co_code
    return checks


def _guard(path: str):
    def not_executable(*_a, **_k):
        raise FrozenModelError(f"stored Python bytecode at {path} is never executed by the live system "
                               "(frozen models are not refitted live)")
    return not_executable


def make_portable(strategy: Strategy, h: int) -> dict:
    """Rebind every by-value function inside `strategy` (in place). Returns a report."""
    lib = {s.name: s for s in rules(h)}
    report = {"python": sys.version.split()[0], "rules_rebuilt": [], "guarded": []}
    seen: set[int] = set()

    def visit(obj, path: str) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, Strategy):
            if obj.model is None and obj.score is not None:
                if obj.name not in lib:
                    raise FrozenModelError(f"rule '{obj.name}' (h={h}) is not in the strategy library")
                if _is_by_value(obj.score):
                    checks = _check_same_rule(obj.name, obj.score, lib[obj.name].score)
                    obj.score = lib[obj.name].score
                    report["rules_rebuilt"].append({"path": f"{path}.score", "rule": obj.name, **checks})
            if obj.model is not None:
                visit(obj.model, f"{path}.model")
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]")
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if _is_by_value(v):
                    obj[k] = _guard(f"{path}[{k!r}]")
                    report["guarded"].append(f"{path}[{k!r}]")
                else:
                    visit(v, f"{path}[{k!r}]")
            return
        if hasattr(obj, "__dict__") and type(obj).__module__.startswith("cryptolab"):
            for k, v in list(vars(obj).items()):
                if _is_by_value(v):
                    setattr(obj, k, _guard(f"{path}.{k}"))
                    report["guarded"].append(f"{path}.{k}")
                else:
                    visit(v, f"{path}.{k}")

    visit(strategy, "strategy")
    return report
