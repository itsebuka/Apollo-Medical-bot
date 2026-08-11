# Apollo Medical Triage System - One-Click Launcher
# ADTC 2026 | Double-click start.bat to run this

$Host.UI.RawUI.WindowTitle = "Apollo - Medical Triage System"

# Resolve all paths relative to THIS script's location
$Root     = $PSScriptRoot
$Python   = "$Root\backend\venv\Scripts\python.exe"
$Backend  = "$Root\backend"
$Frontend = "$Root\frontend"
$ChromaDB = "$Root\backend\chroma_db"

Write-Host ""
Write-Host "  =================================================" -ForegroundColor Green
Write-Host "   APOLLO  |  Medical Triage System  |  ADTC 2026" -ForegroundColor Green
Write-Host "  =================================================" -ForegroundColor Green
Write-Host ""

# STEP 0: Cleanup previous instances to prevent Out of Memory (RAM) crashes
$tcpBackend = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($tcpBackend) {
    Write-Host "  [CLEANUP] Closing old Apollo backend to free up RAM..." -ForegroundColor Yellow
    Stop-Process -Id $tcpBackend.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
$tcpFrontend = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($tcpFrontend) {
    Write-Host "  [CLEANUP] Closing old Apollo frontend..." -ForegroundColor Yellow
    Stop-Process -Id $tcpFrontend.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}
# STEP 1: Check venv
if (-not (Test-Path $Python)) {
    Write-Host "  [ERROR] Python virtual environment not found." -ForegroundColor Red
    Write-Host "          Expected: $Python" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Run this once in your terminal:" -ForegroundColor Yellow
    Write-Host "    cd backend" -ForegroundColor Cyan
    Write-Host "    python -m venv venv" -ForegroundColor Cyan
    Write-Host "    venv\Scripts\python.exe -m pip install chromadb sentence-transformers fastapi uvicorn python-multipart --prefer-binary" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK]   Python venv found." -ForegroundColor Green

# STEP 2: Run ingest if chroma_db is missing or empty
$dbExists = Test-Path $ChromaDB
$dbEmpty  = $dbExists -and ((Get-ChildItem $ChromaDB -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)

if (-not $dbExists -or $dbEmpty) {
    Write-Host "  [SETUP] First run - building vector database..." -ForegroundColor Yellow
    Write-Host "          (downloads embedding model ~90MB once only)" -ForegroundColor Gray
    Write-Host ""
    Push-Location $Backend
    & $Python ingest.py
    $exitCode = $LASTEXITCODE
    Pop-Location
    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "  [ERROR] ingest.py failed (exit code $exitCode)." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host ""
    Write-Host "  [OK]   Vector database ready." -ForegroundColor Green
} else {
    Write-Host "  [OK]   Vector database found - skipping ingest." -ForegroundColor Green
}

Write-Host ""

# STEP 3: Start backend in its own window
Write-Host "  [START] Launching Apollo Backend  ->  http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process "cmd.exe" -ArgumentList "/k cd /d `"$Backend`" && `"$Python`" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"

Start-Sleep -Seconds 2

# STEP 4: Start frontend in its own window
Write-Host "  [START] Launching Apollo Frontend  ->  http://localhost:5173 ..." -ForegroundColor Cyan
Start-Process "cmd.exe" -ArgumentList "/k cd /d `"$Frontend`" && npm run dev"

# STEP 5: Countdown then open browser
Write-Host ""
Write-Host "  [INFO]  Llama-3 8B is loading (4.5 GB model, takes ~15s)..." -ForegroundColor Yellow

for ($i = 15; $i -gt 0; $i--) {
    Write-Host -NoNewline ("`r  [INFO]  Opening browser in {0}s...  " -f $i) -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host ""
Start-Process "http://localhost:5173"

Write-Host "  =================================================" -ForegroundColor Green
Write-Host "   Apollo is running!" -ForegroundColor Green
Write-Host "   Frontend  :  http://localhost:5173" -ForegroundColor Cyan
Write-Host "   Backend   :  http://localhost:8000" -ForegroundColor Cyan
Write-Host "   API Docs  :  http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "  =================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To stop Apollo: close the two cmd windows that opened." -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to close this launcher"
