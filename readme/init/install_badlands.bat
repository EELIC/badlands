@echo off
setlocal

rem === Configuration ===
set ENV_NAME=badlands
set CONDA_PATH=%USERPROFILE%\anaconda3
set WORKSPACE=D:\code-python\badlands
set MAIN_PKG=%WORKSPACE%\badlands-master\badlands
set COMPANION_PKG=%WORKSPACE%\badlands-companion

rem === Activate environment ===
call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment: %ENV_NAME%
    pause
    exit /b 1
)

rem === Install badlands main package ===
cd /d %MAIN_PKG%
echo Installing badlands main package...
pip install -e .

rem === Install companion package ===
cd /d %COMPANION_PKG%
echo Installing badlands companion package...
pip install -e .

rem === Install dependencies ===
echo Installing required packages...
conda install -y numpy=1.21 scipy=1.9.3 pandas h5py scikit-image shapely cmocean jupyterlab
pip install matplotlib==3.3.4 plotly triangle gFlex lavavu piglet pyvirtualdisplay descartes
pip install git+https://github.com/kinnala/meshplex

echo [OK] Installation completed.
pause
