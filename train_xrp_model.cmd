@echo on

:: Set TA-Lib versions
set TALIB_C_VER=0.6.4
set TALIB_PY_VER=0.6.3

:: Set up Python environment
set PYTHON_VERSION=3.10
set PYTHONPATH=%PYTHONPATH%;%CD%

:: Install Python dependencies (excluding TA-Lib for now)
pip install pandas numpy scikit-learn xgboost pyarrow torch tqdm

if errorlevel 1 exit /B 1

:: Download TA-Lib C library
curl -L -o talib-c.zip https://github.com/TA-Lib/ta-lib/archive/refs/tags/v%TALIB_C_VER%.zip
if errorlevel 1 exit /B 1

:: Extract TA-Lib C library
tar -xzvf talib-c.zip
if errorlevel 1 exit /B 1

:: Build TA-Lib C library
setlocal
cd ta-lib-%TALIB_C_VER%
mkdir include\ta-lib
copy /Y include\*.* include\ta-lib
mkdir _build
cd _build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /B 1
nmake /nologo all
if errorlevel 1 exit /B 1
copy /Y /B ta-lib-static.lib ta-lib.lib
endlocal

:: Set environment variables for TA-Lib
set TA_LIBRARY_PATH=%CD%\ta-lib-%TALIB_C_VER%\_build
set TA_INCLUDE_PATH=%CD%\ta-lib-%TALIB_C_VER%\include
set LIB=%TA_LIBRARY_PATH%;%LIB%
set INCLUDE=%TA_INCLUDE_PATH%;%INCLUDE%

:: Install TA-Lib Python package
pip install TA-Lib==%TALIB_PY_VER%
if errorlevel 1 exit /B 1

:: Run the training script
python train_xrp_model.py
if errorlevel 1 exit /B 1

:: Verify output files exist
if not exist "sol_transformer_model.pth" exit /B 1


echo [+] Training completed successfully
