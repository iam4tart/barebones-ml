@echo off
REM clean previous build
if exist build (
    echo Removing old build folder...
    rmdir /s /q build
)

REM compile and install the extension
echo Building libtorch_octree_raii extension...
python setup.py build
python setup.py install

REM clean build folder again after successful build
if exist build (
    echo Cleaning build folder after build...
    rmdir /s /q build
)

echo Done! The .pyd file should now be available in your Python environment.
pause