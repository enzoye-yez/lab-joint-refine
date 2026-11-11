$ErrorActionPreference = 'Stop'
$previousPythonPath = $env:PYTHONPATH
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
$previousEncoding = $env:PYTHONIOENCODING
$previousUtf8 = $env:PYTHONUTF8
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
try {
    $testTemp = Join-Path $PSScriptRoot '.cache\test-temp'
    New-Item -ItemType Directory -Path $testTemp -Force | Out-Null
    $env:TEMP = $testTemp
    $env:TMP = $testTemp
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
    & (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -B -P (Join-Path $PSScriptRoot 'src\runtime_bootstrap.py') --receipt (Join-Path $PSScriptRoot '.cache\verification.json') --module unittest -- discover -s (Join-Path $PSScriptRoot 'tests') -v @args
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
    $env:PYTHONIOENCODING = $previousEncoding
    $env:PYTHONUTF8 = $previousUtf8
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}
