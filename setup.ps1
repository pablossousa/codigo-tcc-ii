$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-Python310Command {
    $candidates = @(
        @{ Command = "py"; Arguments = @("-3.10") },
        @{ Command = "python"; Arguments = @() }
    )

    foreach ($candidate in $candidates) {
        try {
            $version = & $candidate.Command @($candidate.Arguments + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")) 2>$null
            if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.10") {
                return $candidate
            }
        }
        catch {
            continue
        }
    }

    return $null
}

Set-Location $ProjectRoot

if (-not (Test-Path $Requirements)) {
    throw "requirements.txt nao encontrado em: $Requirements"
}

if (-not (Test-Path $VenvPython)) {
    Write-Step "Criando ambiente virtual .venv com Python 3.10"
    $python310 = Get-Python310Command

    if ($null -eq $python310) {
        throw "Python 3.10 nao foi encontrado. Instale o Python 3.10 e execute este script novamente."
    }

    & $python310.Command @($python310.Arguments + @("-m", "venv", $VenvDir))
}
else {
    Write-Step "Ambiente virtual .venv ja existe"
}

Write-Step "Atualizando pip, setuptools e wheel"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Step "Instalando dependencias do requirements.txt"
& $VenvPython -m pip install -r $Requirements

Write-Step "Validando imports principais"
& $VenvPython -c "import torch, facenet_pytorch, cv2, numpy, cryptography, pynput; print('Dependencias OK')"

Write-Host ""
Write-Host "Setup concluido. Para executar o sistema, rode:" -ForegroundColor Green
Write-Host ".\run.ps1" -ForegroundColor Green
