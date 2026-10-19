# ==============================================================================
# scripts/evaluate.ps1: Model Evaluation & Latency Benchmarking
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING BENCHMARK EVALUATION ACROSS ALL MODALITIES" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TRANSFORMERS_NO_TF = "1"
$env:USE_TF = "0"

$pythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    if (Test-Path ".\venv\Scripts\python.exe") {
        $pythonExe = ".\venv\Scripts\python.exe"
    } else {
        $pythonExe = "python"
    }
}

& $pythonExe scripts/evaluate.py
