@echo off
setlocal
set "XCL_NETWORK=0"
set "XCL_LAZY_PSO=1"
set "XCL_LAZY_COMPUTE=1"
set "XCL_HEAP_WINDOWS=1"
set "XCL_DXC_THREADS=2"
set "XCL_PSO_PRELOAD=0"
set "XCL_DXC_DIR=%~dp0"
"%~dp0xcl.exe"
exit /b %errorlevel%
