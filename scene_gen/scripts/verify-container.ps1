param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$managedDocker = Join-Path (Split-Path $projectRoot -Parent) 'tools\Docker\resources\bin'
$oldPath = $env:PATH
$oldBytecode = $env:PYTHONDONTWRITEBYTECODE
$oldEncoding = $env:PYTHONIOENCODING
try {
    if (Test-Path (Join-Path $managedDocker 'docker.exe')) {
        $env:PATH = "$managedDocker;$env:PATH"
    }
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $docker = Get-Command docker -ErrorAction Stop
    $backend = & $docker.Source info --format '{{.OSType}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or $backend -ne 'linux') {
        throw '容器后端尚未就绪。若刚启用系统组件，请保存工作并重启，然后启动 Docker Desktop（容器桌面程序）。'
    }
    if (-not $SkipBuild) {
        & $docker.Source build -f (Join-Path $projectRoot 'containers\asset-worker.Dockerfile') -t harness-asset-worker:0.1 (Join-Path $projectRoot 'containers')
        if ($LASTEXITCODE -ne 0) { throw '构建容器镜像失败，尚未运行工程验证。' }
    }
    & (Join-Path $projectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'verify_backend.py') --backend docker
    if ($LASTEXITCODE -ne 0) { throw '容器工程验证未通过，请查看输出中的检查报告。' }
} finally {
    $env:PATH = $oldPath
    $env:PYTHONDONTWRITEBYTECODE = $oldBytecode
    $env:PYTHONIOENCODING = $oldEncoding
}
