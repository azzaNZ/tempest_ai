@echo off
echo Starting Tempest AI System...

REM Kill existing processes
taskkill /f /im python.exe 2>nul
taskkill /f /im mame.exe 2>nul

REM Start Python server
echo Starting Python server...
cd /d "c:\DataAnnotations\Other\tempest-ai\tempest_ai\Scripts"
start /b python main.py

REM Wait for server
echo Waiting 5 seconds for server to start...
timeout /t 5 /nobreak >nul

REM Start MAME
echo Starting MAME...
cd /d "c:\DataAnnotations\Other\tempest-ai\tempest_ai"
mame tempest1 -autoboot_script Scripts\main.lua -skip_gameinfo

echo Session ended.
pause
