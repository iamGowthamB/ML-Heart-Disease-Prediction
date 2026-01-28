@echo off
echo ======================================================
echo      Heart Disease Prediction System - Launcher
echo ======================================================
echo.
echo [1/2] Checking and Installing Dependencies...
python -m pip install -r requirements.txt --quiet
python -m pip install plotly --quiet

echo.
echo [2/2] Launching Application...
echo.
echo Please wait while the browser opens...
echo.
python -m streamlit run app.py --theme.base="light"

pause
