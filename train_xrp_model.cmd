@echo on


:: Set up Python environment
set PYTHON_VERSION=3.10
set PYTHONPATH=%PYTHONPATH%;%CD%

:: Install Python dependencies (excluding TA-Lib for now)
pip install pandas numpy scikit-learn xgboost pyarrow torch tqdm

if errorlevel 1 exit /B 1


:: Run the training script
set PYTHONUTF8=1
python train_xrp_model.py
if errorlevel 1 exit /B 1

:: Verify output files exist
if not exist "xrp_transformer_model.pth" exit /B 1


echo [+] Training completed successfully
