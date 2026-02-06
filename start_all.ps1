# AlphaTrade System Unified Startup Script
# Run this script to start all components: .\start_all.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "   ALPHATRADE SYSTEM - AUTOMATED STARTUP (VENV)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$VenvPython = ".\venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Error: Virtual Environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "Please ensure you have a 'venv' directory with Python installed." -ForegroundColor Red
    exit 1
}

# 1. Check & Start Redis
Write-Host "`n[1/4] Checking Redis Infrastructure..." -ForegroundColor Yellow
if (Get-Command "docker" -ErrorAction SilentlyContinue) {
    if ((docker ps --format "{{.Names}}" | Select-String "redis") -eq $null) {
        Write-Host "Starting Redis via Docker..."
        docker-compose -f docker/docker-compose.yml up -d redis
        if ($?) { Write-Host "Redis started." -ForegroundColor Green }
        else { Write-Host "Failed to start Redis. Make sure Docker Desktop is running!" -ForegroundColor Red }
    } else {
        Write-Host "Redis is already running." -ForegroundColor Green
    }
} else {
    Write-Host "Docker not found. Skipping Redis auto-start. Ensure Redis is running manually." -ForegroundColor Red
}

# 2. Setup Python Environment
Write-Host "`n[2/4] Verifying Python Dependencies (Venv)..." -ForegroundColor Yellow
& $VenvPython -m pip install -q fastapi uvicorn redis bcrypt sqlalchemy asyncpg psycopg2-binary prometheus_client
if ($?) { Write-Host "Dependencies OK." -ForegroundColor Green }

# 3. Start Dashboard Backend
Write-Host "`n[3/4] Launching Dashboard Backend (Port 8000)..." -ForegroundColor Yellow
Start-Process -FilePath $VenvPython -ArgumentList "main.py dashboard" -WorkingDirectory $PWD -WindowStyle Normal
Write-Host "Backend launched in new window." -ForegroundColor Green

# 4. Start Dashboard Frontend
Write-Host "`n[4/4] Launching Dashboard Frontend (Port 5173)..." -ForegroundColor Yellow
$FrontendDir = Join-Path $PWD "dashboard_ui"
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd '$FrontendDir'; npm run dev" -WindowStyle Normal
Write-Host "Frontend launched in new window." -ForegroundColor Green

# 5. Start Trading Engine (Paper Mode)
Write-Host "`n[+] Launching Trading Engine (Paper Mode)..." -ForegroundColor Yellow
Start-Process -FilePath $VenvPython -ArgumentList "main.py trade --mode paper" -WorkingDirectory $PWD -WindowStyle Normal
Write-Host "Trading Engine launched in new window." -ForegroundColor Green

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "   SYSTEM STARTUP COMPLETE" -ForegroundColor Cyan
Write-Host "   Access Dashboard at: http://localhost:5173" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
