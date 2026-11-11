$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$oldBytecode = $env:PYTHONDONTWRITEBYTECODE
$oldEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    & (Join-Path $projectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'verify_backend.py') --backend local @args
    if ($LASTEXITCODE -ne 0) { throw '本地工程验证未通过，请查看输出中的检查报告。' }
} finally {
    $env:PYTHONDONTWRITEBYTECODE = $oldBytecode
    $env:PYTHONIOENCODING = $oldEncoding
}
