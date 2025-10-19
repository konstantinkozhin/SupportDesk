<#
NGROK_RUNNER.ps1

Helper to start the local app and expose it via ngrok (Windows PowerShell).
- Does NOT modify your project code.
- Uses ngrok local agent (must be installed separately if not present).

Usage examples:
  # Option A: pass token on the command line
  .\NGROK_RUNNER.ps1 -NgrokToken "32Pf3JDFeJTHcRoi0NkCqgUVV68_48P3gyynHQFuwqp8QJpC4"

  # Option B: set environment variable NGROK_AUTHTOKEN and call script without args
  $env:NGROK_AUTHTOKEN = '32Pf3JDFeJTHcRoi0NkCqgUVV68_48P3gyynHQFuwqp8QJpC4'
  .\NGROK_RUNNER.ps1

Notes:
- Script assumes the app listens on localhost:8000. If your app runs on a different port, pass -Port <port>.
- The script will try to run the Python uvicorn server (python -m uvicorn app.main:app --reload --port <port>). If you prefer using your existing batch (start_server.bat), edit the script accordingly or pass -UseBatch switch.
- Keep your ngrok authtoken secret. If you share this machine, prefer setting the token in an environment variable rather than embedding it in files.
#>

param(
    [string]$NgrokToken = $env:NGROK_AUTHTOKEN,
    [int]$Port = 8000,
    [switch]$UseBatch  # when set, tries to run start_server.bat instead of uvicorn
)

function Abort([string]$msg){ Write-Host $msg -ForegroundColor Red; exit 1 }

if (-not $NgrokToken) {
    Abort "Ngrok authtoken not provided. Set environment variable NGROK_AUTHTOKEN or pass -NgrokToken '<token>'"
}

# Check ngrok availability
$ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ngrokCmd) {
    Write-Host "ngrok executable not found in PATH." -ForegroundColor Yellow
    Write-Host "Please install ngrok from https://ngrok.com/download and add it to your PATH, or run it from its folder." -ForegroundColor Yellow
    Write-Host "You can also place ngrok.exe into the repository root." -ForegroundColor Yellow
    Pause
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $scriptDir

# Start the local server
if ($UseBatch) {
    if (Test-Path (Join-Path $scriptDir 'start_server.bat')) {
        Write-Host "Starting server with start_server.bat..." -ForegroundColor Green
        Start-Process -FilePath (Join-Path $scriptDir 'start_server.bat') -WorkingDirectory $scriptDir
    } else {
        Abort "start_server.bat not found in repository root. Use uvicorn mode or add the batch file."
    }
} else {
    Write-Host "Starting uvicorn server on port $Port..." -ForegroundColor Green
    # start a new PowerShell window so the server stays running after this script prints the URL
    $uvicornCmd = "python -m uvicorn app.main:app --reload --host 127.0.0.1 --port $Port"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit","-Command",$uvicornCmd -WorkingDirectory $scriptDir
}

# Give server time to come up
Write-Host "Waiting a few seconds for the local server to start..."
Start-Sleep -Seconds 3

# Configure ngrok with authtoken (idempotent)
Write-Host "Configuring ngrok authtoken..." -ForegroundColor Green
& ngrok authtoken $NgrokToken | Out-Null

# Start ngrok tunnel
Write-Host "Starting ngrok tunnel (http -> localhost:$Port)..." -ForegroundColor Green
Start-Process -FilePath "ngrok" -ArgumentList "http $Port --log=stdout" -WindowStyle Hidden

# Wait for ngrok local API to respond and print public URL
$api = 'http://127.0.0.1:4040/api/tunnels'
$attempt = 0
$tunnelUrl = $null
while ($attempt -lt 20 -and -not $tunnelUrl) {
    try {
        $attempt++
        $resp = Invoke-RestMethod -Uri $api -Method Get -ErrorAction Stop
        if ($resp.tunnels -and $resp.tunnels.Count -gt 0) {
            # pick the first http tunnel
            $httpTunnel = $resp.tunnels | Where-Object { $_.proto -eq 'http' } | Select-Object -First 1
            if ($httpTunnel) { $tunnelUrl = $httpTunnel.public_url }
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if ($tunnelUrl) {
    Write-Host "ngrok public URL: $tunnelUrl" -ForegroundColor Cyan
    Write-Host "You can share this URL to reach your local app (tunnel will remain active while ngrok process runs)." -ForegroundColor Gray
} else {
    Write-Host "Failed to read ngrok tunnels from local API (http://127.0.0.1:4040)." -ForegroundColor Red
    Write-Host "Open http://127.0.0.1:4040 in your browser to view active tunnels and debug." -ForegroundColor Yellow
}

Pop-Location

# End of script
