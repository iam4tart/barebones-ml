@echo off
echo cleaning previous build...
rd /s /q build 2>nul
del /q barebones\libs\*.pyd 2>nul
del /q barebones\libs\*.so 2>nul

echo building extensions...
python setup.py build_ext --inplace

echo.
echo verifying output in barebones\libs\:
dir barebones\libs\*.pyd 2>nul || echo no .pyd files found - build may have failed

echo.
echo testing import...
python -c "import torch; import barebones; print('ok:', dir(barebones))"