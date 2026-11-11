# 启动入口；所有本地产物限定在当前新项目。
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '缺少项目环境，请先按照 README.md（使用说明）创建 .venv（虚拟环境）。'
}
$previousPythonPath = $env:PYTHONPATH
$previousBytecode = $env:PYTHONDONTWRITEBYTECODE
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    if ($args.Count -gt 0 -and $args[0] -eq 'prepare') {
        $prepareArgs = @($args | Select-Object -Skip 1)
        & $pythonPath -B -P (Join-Path $projectRoot 'src\runtime_bootstrap.py') --module asset_prep.prepare_cli -- @prepareArgs
    } elseif ($args.Count -gt 0 -and $args[0] -eq 'operate') {
        $operationArgs = @($args | Select-Object -Skip 1)
        & $pythonPath -B -P (Join-Path $projectRoot 'src\runtime_bootstrap.py') --module robot_operation -- @operationArgs
    } elseif ($args.Count -gt 0 -and $args[0] -eq 'preview') {
        $previewArgs = @($args | Select-Object -Skip 1)
        & $pythonPath -m scene_generation.preview @previewArgs
    } elseif ($args.Count -gt 0 -and $args[0] -eq 'scene') {
        $sceneArgs = @($args | Select-Object -Skip 1)
        & $pythonPath -B -P (Join-Path $projectRoot 'src\runtime_bootstrap.py') --module scene_generation -- @sceneArgs
    } elseif ($args.Count -gt 0 -and $args[0] -eq 'generate') {
        $generationArgs = @($args | Select-Object -Skip 1)
        & $pythonPath -m asset_generation @generationArgs
    } else {
        & $pythonPath -m asset_prep @args
    }
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecode
    $env:PYTHONIOENCODING = $previousEncoding
}
