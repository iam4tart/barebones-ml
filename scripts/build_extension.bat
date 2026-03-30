@echo off
REM clean previous build
if exist build (
    echo Removing old build folder...
    rmdir /s /q build
)

REM compile and install the extension
echo Building barebones extension...
python setup.py build_ext --inplace

REM clean build folder again after successful build
if exist build (
    echo Cleaning build folder after build...
    rmdir /s /q build
)

echo Done! Run 'from barebones import octree' in Python.
pause