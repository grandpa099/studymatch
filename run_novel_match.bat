@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -m venv .venv
  if errorlevel 1 python -m venv .venv
  call ".venv\Scripts\activate"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call ".venv\Scripts\activate"
)

python app.py
pause
