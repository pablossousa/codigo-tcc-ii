$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AppFile = Join-Path $ProjectRoot "face_recognition_app.py"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Ambiente .venv nao encontrado. Execute primeiro:" -ForegroundColor Yellow
    Write-Host ".\setup.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $AppFile)) {
    throw "Arquivo principal nao encontrado: $AppFile"
}
w
Set-Location $ProjectRoot
& $VenvPython $AppFile @args
exit $LASTEXITCODE
