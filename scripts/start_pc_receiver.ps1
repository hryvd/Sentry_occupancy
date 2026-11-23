<#
.SYNOPSIS
Start the local SENTRY Lot B2, B3 & B4 HTTP receiver and a Cloudflare Quick Tunnel in one command.

.DESCRIPTION
Starts `scripts/run_lots_b2_b4_local.py --serve` on 127.0.0.1:$Port with token authentication,
downloads standalone `cloudflared.exe` (no admin required) if not already installed, launches an
unthrottled `*.trycloudflare.com` HTTPS tunnel, and prints the tunnel URL, token, and pairing
query string for the Vercel UI. Press Ctrl+C to stop both processes cleanly.
#>
param(
    [int]$Port = 8080,
    [string]$Token = "sentry-field-2026",
    [string]$HostAddr = "127.0.0.1",
    [string]$CloudflaredPath = "$env:USERPROFILE\.local\bin\cloudflared.exe"
)

$ErrorActionPreference = "Stop"

$existingListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existingListener) {
    $pids = ($existingListener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    Write-Error "Port $Port is already listening (PID: $pids). Stop the existing receiver first before starting a new one."
    exit 1
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$serverScript = Join-Path $repoRoot "scripts\run_lots_b2_b4_local.py"
if (-not (Test-Path $serverScript)) {
    Write-Error "Could not locate receiver script: $serverScript"
    exit 1
}

if (-not (Test-Path $CloudflaredPath) -or (Get-Item $CloudflaredPath).Length -lt 1000000) {
    $cfDir = Split-Path -Parent $CloudflaredPath
    New-Item -ItemType Directory -Force -Path $cfDir | Out-Null
    Write-Host "Downloading standalone cloudflared to $CloudflaredPath ..." -ForegroundColor Cyan
    Invoke-WebRequest `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $CloudflaredPath `
        -UseBasicParsing
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python executable was not found in PATH."
    exit 1
}

$tempDir = [System.IO.Path]::GetTempPath()
$cfStdOut = Join-Path $tempDir "sentry_cloudflared_out_$PID.log"
$cfStdErr = Join-Path $tempDir "sentry_cloudflared_err_$PID.log"
$pyProc = $null
$cfProc = $null

try {
    Write-Host "Starting local SENTRY receiver on http://${HostAddr}:${Port} ..." -ForegroundColor Cyan
    $pyArgs = @(
        "`"$serverScript`"",
        "--serve",
        "--skip-initial-run",
        "--host", $HostAddr,
        "--port", "$Port"
    )
    if (-not [string]::IsNullOrWhiteSpace($Token)) {
        $pyArgs += @("--token", "`"$Token`"")
    }

    $pyProc = Start-Process -FilePath $pythonCmd.Source `
        -ArgumentList ($pyArgs -join " ") `
        -WorkingDirectory $repoRoot `
        -NoNewWindow `
        -PassThru

    Start-Sleep -Seconds 2
    if ($pyProc.HasExited) {
        Write-Error "Local receiver exited early with code $($pyProc.ExitCode)."
        exit 1
    }

    Write-Host "Starting Cloudflare Quick Tunnel (trycloudflare.com) -> http://${HostAddr}:${Port} ..." -ForegroundColor Cyan
    $cfProc = Start-Process -FilePath $CloudflaredPath `
        -ArgumentList "tunnel --url http://${HostAddr}:${Port} --no-autoupdate" `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $cfStdOut `
        -RedirectStandardError $cfStdErr `
        -NoNewWindow `
        -PassThru

    $tunnelUrl = $null
    for ($i = 0; $i -lt 45; $i++) {
        if ($cfProc.HasExited) {
            $errText = if (Test-Path $cfStdErr) { Get-Content $cfStdErr -Raw -ErrorAction SilentlyContinue } else { "" }
            Write-Error "cloudflared exited early (code $($cfProc.ExitCode)): $errText"
            exit 1
        }
        $combinedLog = ""
        if (Test-Path $cfStdOut) { $combinedLog += (Get-Content $cfStdOut -Raw -ErrorAction SilentlyContinue) }
        if (Test-Path $cfStdErr) { $combinedLog += "`n" + (Get-Content $cfStdErr -Raw -ErrorAction SilentlyContinue) }
        if ($combinedLog -match "(https://[a-z0-9-]+\.trycloudflare\.com)") {
            $tunnelUrl = $Matches[1]
            break
        }
        Start-Sleep -Seconds 1
    }

    if (-not $tunnelUrl) {
        Write-Error "Timed out waiting for trycloudflare.com URL. Check logs at $cfStdErr"
        exit 1
    }

    $encUrl = [System.Uri]::EscapeDataString($tunnelUrl)
    $pairingQs = "?api=$encUrl"
    if (-not [string]::IsNullOrWhiteSpace($Token)) {
        $pairingQs += "&token=$([System.Uri]::EscapeDataString($Token))"
    }

    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host "  SENTRY PC RECEIVER & CLOUDFLARE TUNNEL ACTIVE" -ForegroundColor Green
    Write-Host "  Local Receiver : http://${HostAddr}:${Port}" -ForegroundColor White
    Write-Host "  HTTPS Tunnel   : $tunnelUrl" -ForegroundColor Cyan
    Write-Host "  Access Token   : $(if ($Token) { $Token } else { '(disabled)' })" -ForegroundColor Yellow
    Write-Host "  Pairing Suffix : $pairingQs" -ForegroundColor Gray
    Write-Host "  Press Ctrl+C to stop both the receiver and tunnel." -ForegroundColor Gray
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host ""

    while (-not $pyProc.HasExited -and -not $cfProc.HasExited) {
        Start-Sleep -Seconds 2
    }
}
finally {
    if ($cfProc -and -not $cfProc.HasExited) {
        Stop-Process -Id $cfProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($pyProc -and -not $pyProc.HasExited) {
        Stop-Process -Id $pyProc.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -Path $cfStdOut, $cfStdErr -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped SENTRY PC receiver and Cloudflare tunnel cleanly." -ForegroundColor Yellow
}
