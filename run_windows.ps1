param([switch]$TrainDemo)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
function Run-Python {
    & $ProjectPython @args
    if ($LASTEXITCODE -ne 0) { throw 'Python command failed; see error above' }
}
Run-Python -m pip install -r requirements.txt
Run-Python build_native.py
Run-Python -m pytest -q
Run-Python -m experiments.reconstruct --demo
if ($TrainDemo) {
    Run-Python -m pip install -r requirements-experiments.txt
    Run-Python -m experiments.make_demo
    $RunName = 'results/training_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
    Run-Python -m experiments.train --manifest data/synthetic/manifest.json --output $RunName --epochs 20
}
Write-Host 'Finished. These demos verify code; they do not measure real-world person segmentation.'
