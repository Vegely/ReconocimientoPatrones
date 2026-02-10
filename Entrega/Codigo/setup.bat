@echo off
echo Installing required libraries...
pip install gdown

echo.
echo Starting Data Download...
python download_data.py

echo.
echo Done! You can close this window.
pause