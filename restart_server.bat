@echo off
echo Stopping test bot...
taskkill /F /FI "WINDOWTITLE eq test_bot_simple*" 2>nul
timeout /t 2 /nobreak >nul

echo Starting main server...
cd /d "%~dp0"
call .venv\Scripts\activate
python -m uvicorn app.main:app --reload
