@echo off
:: ==============================================================================
:: 🎛️ TEST DASHBOARD: CHANGE YOUR PARAMETERS HERE
:: ==============================================================================

set SEED=1
set EXP_NAME=test_local_01
set TOTAL_STEPS=20000
set ORACLE_COST=0.2

:: To test the baseline (no oracle), leave the next line as --no-oracle
:: To test the oracle, change it to just: set NO_ORACLE_FLAG=
:: set NO_ORACLE_FLAG=--no-oracle

:: ==============================================================================
:: 🚀 LAUNCH LOGIC (Do not change below this line)
:: ==============================================================================

echo Preparing local Windows test job...
echo  - Experiment: %EXP_NAME%
echo  - Seed: %SEED%
echo  - Total Steps: %TOTAL_STEPS%
echo  - Oracle Cost: %ORACLE_COST%
echo  - Oracle Flag: %NO_ORACLE_FLAG%
echo.

:: Create folders if they don't exist
if not exist logs mkdir logs
if not exist figures mkdir figures
if not exist checkpoints mkdir checkpoints

echo Starting training...
python gym_sokoban\train.py --exp-name %EXP_NAME% --seed %SEED% --total-timesteps %TOTAL_STEPS% --oracle-cost %ORACLE_COST% %NO_ORACLE_FLAG%

echo.
echo Local test finished!