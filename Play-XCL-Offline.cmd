@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Play-XCL-Offline.ps1"
set "XCL_PLAY_EXIT=%ERRORLEVEL%"
if not "%XCL_PLAY_EXIT%"=="0" pause
exit /b %XCL_PLAY_EXIT%
