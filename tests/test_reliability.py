"""Supervisor restarts hung workers, honours the stop file; outages are recorded as events."""
import threading
import time

from cryptolab.config import load_config
from cryptolab.live import supervisor as sup
from cryptolab.storage.db import Store


def _cfg(tmp_path):
    cfg = load_config()
    for k, v in {"data_dir": "data", "db_file": "data/db.sqlite", "ledger_dir": "data/ledger", "log_dir": "logs"}.items():
        cfg["paths"][k] = str(tmp_path / v)
    return cfg


def test_supervisor_restarts_hung_worker_and_stops_cleanly(tmp_path):
    cfg = _cfg(tmp_path)
    # A "worker" that never writes a heartbeat (hung) -> must be killed and restarted.
    t = threading.Thread(target=sup.run_supervisor, kwargs={
        "cfg": cfg, "hang_after_s": 2, "worker_args": ["-c", "import time; time.sleep(600)"]})
    t.start()
    time.sleep(20)  # hang detected ~3 s, restart after a 10 s backoff
    sup.stop_file(cfg).touch()
    t.join(90)
    assert not t.is_alive()
    kinds = [e["kind"] for e in Store(tmp_path / "data/db.sqlite", tmp_path / "data/ledger").rows("events")]
    assert kinds[0] == "supervisor_start" and kinds[-1] == "supervisor_stop"
    assert kinds.count("worker_hung") >= 1 and kinds.count("worker_start") >= 2
    assert not sup.stop_file(cfg).exists()


def test_network_outage_is_recorded(tmp_path):
    from cryptolab.live.scheduler import LiveSystem
    cfg = _cfg(tmp_path)
    s = LiveSystem.__new__(LiveSystem)
    s.cfg, s.outage_since, s.outage_error = cfg, None, ""
    s.store = Store(tmp_path / "data/db.sqlite", tmp_path / "data/ledger")

    class _BN:
        fail = True

        def server_time(self):
            if self.fail:
                raise ConnectionError("no route to host")
            return 0
    s.bn = _BN()
    assert s.check_connectivity() is False
    assert s.check_connectivity() is False          # still down: only ONE start event
    s.bn.fail = False
    assert s.check_connectivity() is True
    ev = [e["kind"] for e in Store(tmp_path / "data/db.sqlite", tmp_path / "data/ledger").rows("events")]
    assert ev == ["network_outage_start", "network_outage_end"]
    assert "network outage" in s.store.get_runtime("last_error:A_10min")[0]


def test_stale_pid_file_does_not_block_start(tmp_path):
    """A pid file left behind by a dead supervisor (pid reused by another process) must not block start."""
    import os
    cfg = _cfg(tmp_path)
    (sup.run_dir(cfg) / "supervisor.pid").write_text(str(os.getpid()))  # a LIVE but unrelated pid
    st = sup.status(cfg)
    assert st["supervisor_alive"] is False and st["worker_alive"] is False
    assert sup.lock_held(sup.run_dir(cfg) / "supervisor.lock") is False
