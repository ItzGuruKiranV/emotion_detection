# ==============================================================================
# scripts/setup.ps1: Complete Windows Automated Setup Script
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "MULTIMODAL EMOTION ANALYSIS PLATFORM - WINDOWS SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Python version
Write-Host "`n[1/5] Verifying Python Environment..." -ForegroundColor Yellow
$pythonCmd = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
}

& $pythonCmd --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10, 3.11, 3.12, or 3.13."
    exit 1
}

# 2. Virtual Environment Creation
Write-Host "`n[2/5] Setting up Virtual Environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    & $pythonCmd -m venv .venv
    Write-Host "Created new virtual environment in .venv" -ForegroundColor Green
} else {
    Write-Host "Existing virtual environment detected in .venv" -ForegroundColor Green
}

# 3. Environment Variables & Pip Install
Write-Host "`n[3/5] Installing and Updating Dependencies..." -ForegroundColor Yellow
$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

# If .venv does not have Scripts yet (e.g. using existing venv/)
if (-not (Test-Path $venvPython)) {
    if (Test-Path ".\venv\Scripts\python.exe") {
        $venvPython = ".\venv\Scripts\python.exe"
        $venvPip = ".\venv\Scripts\pip.exe"
    }
}

& $venvPip install --upgrade pip
& $venvPip install -r requirements.txt

# 4. Generate Sample Assets
Write-Host "`n[4/5] Generating Offline Sample Assets..." -ForegroundColor Yellow
& $venvPython scripts/create_sample_assets.py

# 5. Initialize Database and Self-Check
Write-Host "`n[5/5] Initializing Database & Executing Self-Test..." -ForegroundColor Yellow
& $venvPython -c "from src.database.session import init_db; init_db(); print('Database schema ready!')"

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "To run the application:" -ForegroundColor White
Write-Host "   .\scripts\run.ps1" -ForegroundColor Cyan
Write-Host "To run the test suite:" -ForegroundColor White
Write-Host "   .\scripts\test.ps1" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Green
