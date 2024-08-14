@echo off
setlocal

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo requirements.txt not found. Please make sure it exists in the current directory.
    pause
    exit /B 1
)

REM Create a virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

REM Launch the application
echo Launching the application...
python app.py

REM Deactivate the virtual environment
deactivate

endlocal
pause