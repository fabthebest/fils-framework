@echo off
echo ========================================
echo  FILS Framework - Push to GitHub
echo ========================================
echo.

cd /d "C:\Users\fabri\OneDrive\Documents\Claude\Projects\Fils-Aimé Méthode\fils-aime-method"

echo Initializing git repository...
git init

echo.
echo Adding all files...
git add .

echo.
echo Creating initial commit...
git commit -m "Initial commit: FILS Framework v0.1.0 - 44 concept cards, 3 AI agents, mastery validation"

echo.
echo Setting branch to main...
git branch -M main

echo.
echo Adding GitHub remote...
git remote add origin https://github.com/fabthebest/fils-framework.git

echo.
echo Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo  Done! Check github.com/fabthebest/fils-framework
echo ========================================
pause
