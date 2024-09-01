@echo off
:: Batch script to create folder structure for Flask Emotion Detection App

:: Root folder
mkdir emotions_app
cd emotions_app

:: Create static and css folders
mkdir static
mkdir static\css

:: Create templates folder
mkdir templates

:: Go back to root directory
cd ..

echo Folder structure created successfully!
pause
