# Start both servers in independent windows — run once per work session
$root = $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; uvicorn backend.main:app --reload --port 8001"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"

Write-Host "Servers starting in separate windows."
Write-Host "  Backend:  http://localhost:8001"
Write-Host "  Frontend: http://localhost:3000"
