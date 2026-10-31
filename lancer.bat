@echo off
setlocal
title Recherche physique quantique
cd /d "%~dp0"

rem Environnement Python 3.11 commun a tous les scripts du dossier
set "PY=%~dp0env_311\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERREUR] Python introuvable : "%PY%"
    pause
    exit /b 1
)

:menu
cls
echo ============================================================
echo              RECHERCHE PHYSIQUE QUANTIQUE
echo ============================================================
echo.
echo   QISKIT - calcul quantique
echo     1. Superposition quantique, 1 qubit ........ test.py
echo     2. Intrication, etat de Bell, 2 qubits ..... test_qiskit_2qubits.py
echo     3. Teleportation quantique, 3 qubits ....... teleportation_quantique.py
echo.
echo   SIMULATIONS PHYSIQUES
echo     4. Fluide 3D, 10 000 particules - Taichi ... fluide_taichi.py
echo     5. Test Taichi / GPU ....................... test_taichi.py
echo     6. Test MuJoCo - boite qui tombe ........... test_mujoco.py
echo.
echo   OUTILS
echo     7. Installer Qiskit dans l'environnement
echo     8. Ouvrir un terminal avec l'environnement active
echo     0. Quitter
echo.
set "CHOIX="
set /p "CHOIX=Ton choix : "

if "%CHOIX%"=="1" call :qiskit test.py
if "%CHOIX%"=="2" call :qiskit test_qiskit_2qubits.py
if "%CHOIX%"=="3" call :qiskit teleportation_quantique.py
if "%CHOIX%"=="4" call :run fluide_taichi.py
if "%CHOIX%"=="5" call :run test_taichi.py
if "%CHOIX%"=="6" call :run test_mujoco.py
if "%CHOIX%"=="7" (
    call :install
    pause
)
if "%CHOIX%"=="8" start "env_311" cmd /k ""%~dp0env_311\Scripts\activate.bat""
if "%CHOIX%"=="0" exit /b 0
goto menu


rem ---- Lance un script avec l'environnement commun --------------------------
:run
echo.
echo === %~1 ===
echo.
"%PY%" "%~1"
echo.
echo ------------------------------------------------------------
pause
goto :eof


rem ---- Lance un script Qiskit (installe Qiskit au besoin) -------------------
:qiskit
"%PY%" -c "import qiskit, qiskit_aer" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Qiskit n'est pas encore installe dans env_311 : installation automatique...
    call :install
    "%PY%" -c "import qiskit, qiskit_aer" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ERREUR] Qiskit n'a pas pu etre installe. Lis les messages ci-dessus.
        pause
        goto :eof
    )
)
call :run "%~1"
goto :eof


rem ---- Installe qiskit + qiskit-aer dans env_311 ----------------------------
:install
echo.
echo Installation de qiskit et qiskit-aer dans env_311...
where uv >nul 2>&1
if errorlevel 1 (
    "%PY%" -m ensurepip --upgrade
    "%PY%" -m pip install qiskit qiskit-aer
) else (
    uv pip install --python "%PY%" qiskit qiskit-aer
)
goto :eof
