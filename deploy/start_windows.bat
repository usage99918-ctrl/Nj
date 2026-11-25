@echo off
rem Manual start without Task Scheduler (window can be closed afterwards).
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m cryptolab start
".venv\Scripts\python.exe" -m cryptolab status
pause
