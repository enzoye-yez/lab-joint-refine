# 顺序场景实验入口；默认仅准备清单，--execute 才调用模型。
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw '缺少项目 Python（解释器）环境。' }
$previousPythonPath = $env:PYTHONPATH
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    & $pythonPath (Join-Path $PSScriptRoot 'run_scene_batch.py') @args
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
    $env:PYTHONIOENCODING = $previousEncoding
}
