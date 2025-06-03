@echo off
for /l %%x in (1,1,16) do (
    start /b mame tempest1 -skip_gameinfo -autoboot_script c:\DataAnnotations\Other\tempest_ai\Scripts\main.lua -nothrottle -sound none -window -frameskip 9 >nul