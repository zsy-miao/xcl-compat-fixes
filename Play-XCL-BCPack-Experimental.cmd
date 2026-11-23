@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Play-XCL-BCPack-Experimental.ps1"
if errorlevel 1 pause
