<#
.SYNOPSIS
    Install Meet on Windows: core app, `meet` command on PATH, then `meet setup`.

.DESCRIPTION
    Run from a checkout (or an extracted release zip):

        powershell -ExecutionPolicy Bypass -File scripts\install.ps1

    Creates:
        %USERPROFILE%\.meet\core      the Meet app (its own Python environment)
        %USERPROFILE%\.meet\bin       meet.cmd, added to your user PATH
    then runs `meet setup`, which builds the listener runtime and downloads
    the models into %USERPROFILE%\.meet.

    Safe to re-run: it upgrades in place. Nothing is installed system-wide and
    no administrator rights are needed.
#>
[CmdletBinding()]
param(
    [string]$Source = "",
    [string]$Model = "small.en",
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"
# Resolved here, not as a param default: Windows PowerShell 5.1 leaves
# $PSScriptRoot empty while evaluating param() defaults under -File.
if (-not $Source) {
    $Here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
    $Source = Split-Path -Parent $Here
}
if (Test-Path $Source) { $Source = (Resolve-Path $Source).Path }
$MeetHome = if ($env:MEET_HOME) { $env:MEET_HOME } else { Join-Path $env:USERPROFILE ".meet" }
$Core = Join-Path $MeetHome "core"
$Bin = Join-Path $MeetHome "bin"

function Say($text) { Write-Host "  $text" }
function Fail($text, $fix) {
    Write-Host "  x $text" -ForegroundColor Red
    if ($fix) { Write-Host "      $fix" }
    exit 1
}

Write-Host "MEET INSTALL"

# ── Python 3.12+ ────────────────────────────────────────────────────────
# Prefer the py launcher, which finds any installed version regardless of PATH
# order; fall back to `python` on PATH. The Microsoft Store alias for python.exe
# is a stub that opens the Store, so the version check below rejects it.
$Python = $null
foreach ($candidate in @("py -3.13", "py -3.12", "python")) {
    try {
        $parts = $candidate -split " "
        $exe = $parts[0]
        $rest = @($parts | Select-Object -Skip 1)
        $version = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        # Only the versions the lock is verified for (see scripts/check_wheels.py).
        if ($LASTEXITCODE -eq 0 -and $version -in @("3.12", "3.13")) {
            $Python = $parts
            break
        }
    } catch { }
}
if (-not $Python) {
    Fail "Python 3.12 or 3.13 was not found" "Install it with:  winget install Python.Python.3.12   then open a new terminal and re-run this script."
}
$PyExe = $Python[0]
$PyArgs = @($Python | Select-Object -Skip 1)
Say "+ Python $version"

if (-not (Test-Path (Join-Path $Source "pyproject.toml"))) {
    Fail "no Meet source at $Source" "Run this script from inside the Meet folder, or pass -Source <folder>."
}

# ── core environment ────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $MeetHome, $Bin | Out-Null
$CorePy = Join-Path $Core "Scripts\python.exe"
if (-not (Test-Path $CorePy)) {
    & $PyExe @PyArgs -m venv $Core
    if ($LASTEXITCODE -ne 0) { Fail "could not create $Core" "Check that $MeetHome is writable." }
}
$pip = @("-m", "pip", "--disable-pip-version-check", "install", "--no-input", "--quiet")
& $CorePy @pip --require-hashes --no-deps -r (Join-Path $Source "requirements.lock")
if ($LASTEXITCODE -ne 0) { Fail "installing Meet's dependencies failed" "Check your internet connection and re-run." }
& $CorePy @pip --no-deps --force-reinstall $Source
if ($LASTEXITCODE -ne 0) { Fail "installing Meet failed" "Re-run with -Verbose, or report the error above." }
Say "+ Meet app in $Core"

# ── `meet` on PATH ──────────────────────────────────────────────────────
$Shim = Join-Path $Bin "meet.cmd"
Set-Content -Path $Shim -Encoding ASCII -Value "@echo off`r`n`"$Core\Scripts\meet.exe`" %*"
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $UserPath) { $UserPath = "" }
if (-not (($UserPath -split ";") -contains $Bin)) {
    $NewPath = if ($UserPath) { "$UserPath;$Bin" } else { $Bin }
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    Say "+ added $Bin to your PATH (new terminals will see it)"
}
$env:Path = "$env:Path;$Bin"

# ── listener runtime, models, checks ────────────────────────────────────
if ($SkipSetup) {
    Say "done. Next: meet setup"
    exit 0
}
& $Shim setup --model $Model
exit $LASTEXITCODE
