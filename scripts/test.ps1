# ==============================================================================
# scripts/test.ps1: Automated Pytest Test Runner
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING AUTOMATED TEST SUITE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TRANSFORMERS_NO_TF = "1"
$env:USE_TF = "0"

$pytestExe = ".\.venv\Scripts\pytest.exe"
if (-not (Test-Path $pytestExe)) {
    if (Test-Path ".\venv\Scripts\pytest.exe") {
        $pytestExe = ".\venv\Scripts\pytest.exe"
    } else {
        $pytestExe = "pytest"
    }
}

& $pytestExe -v tests/
