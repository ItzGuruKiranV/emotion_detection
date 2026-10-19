# ==============================================================================
# scripts/run.ps1: Launch Streamlit Application
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "LAUNCHING MULTIMODAL EMOTION ANALYSIS PLATFORM" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Set memory-safety environment flags
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TRANSFORMERS_NO_TF = "1"
$env:USE_TF = "0"

# Locate virtual environment streamlit
$streamlitExe = ".\.venv\Scripts\streamlit.exe"
if (-not (Test-Path $streamlitExe)) {
    if (Test-Path ".\venv\Scripts\streamlit.exe") {
        $streamlitExe = ".\venv\Scripts\streamlit.exe"
    } else {
        $streamlitExe = "streamlit"
    }
}

Write-Host "Starting Streamlit Server on http://localhost:8501 ..." -ForegroundColor Green
& $streamlitExe run app.py
