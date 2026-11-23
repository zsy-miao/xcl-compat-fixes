@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Update-XCL.ps1" %*
set "XCL_UPDATE_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %XCL_UPDATE_EXIT%
