@echo off
REM One-click: update the dataset from the live source and regenerate both
REM prediction versions (live + pre-tournament). Double-click this file.
cd /d "%~dp0"

echo ============================================================
echo  FIFA World Cup 2026 - update and run
echo ============================================================
echo.
echo [1/2] Updating dataset from the live source (martj42)...
python update_data.py
echo.
echo [2/2] Running predictions (live + pre-tournament)...
python run_pipeline.py both
echo.
echo ============================================================
echo  Done! Open outputs\dashboard.html in your browser.
echo  (comparison page: compare.html)
echo ============================================================
pause
