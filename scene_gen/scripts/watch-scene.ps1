param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')]
    [string]$Run,
    [switch]$Watch,
    [ValidateRange(1, 60)]
    [int]$IntervalSeconds = 5
)

# 独立只读监视器：不导入生成模块，不调用模型，不获取实验写锁。
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$runRoot = Join-Path (Join-Path $projectRoot 'runs') $Run
$statePath = Join-Path $runRoot 'state.json'
$chinaZone = [TimeZoneInfo]::FindSystemTimeZoneById('China Standard Time')

function Get-StatusLabel([string]$Value) {
    switch ($Value) {
        'running' { '进行中（账本状态）' }
        'candidate_ready' { '候选就绪' }
        'assembly_ready' { '可装配，仍有待验证项' }
        'review_pending' { '已有模型，等待按新阶段规则重审' }
        'static_candidate_ready' { '静态候选就绪' }
        'static_preview_ready' { '静态布局预览就绪，后续状态及配合仍待验证' }
        'needs_review' { '需要复核' }
        'needs_environment' { '环境需要处理' }
        'failed' { '未通过' }
        'paused' { '已暂停' }
        'asset_repair_budget_exhausted' { '单资产修复次数耗尽' }
        'scene_repair_budget_exhausted' { '场景修复次数耗尽' }
        default { $Value }
    }
}

function Show-SceneSnapshot {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        throw "尚未找到运行账本：$statePath"
    }
    $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $state.scene) { throw '此目录不是场景生成运行。' }
    $now = [DateTimeOffset]::UtcNow
    $stamp = [TimeZoneInfo]::ConvertTime($now, $chinaZone).ToString('HH:mm:ss')
    $age = [math]::Max(0, [int]($now.UtcDateTime - (Get-Item -LiteralPath $statePath).LastWriteTimeUtc).TotalSeconds)
    $jobs = @($state.scene.plan.jobs)
    $done = 0
    foreach ($job in $jobs) {
        $entry = $state.scene.jobs.($job.id)
        if ($entry.status -in @('candidate_ready', 'assembly_ready') -and -not $entry.dirty) { $done++ }
    }
    $calls = @($state.calls | Where-Object { $_.kind -in @('model', 'model_search') })
    $pending = @($calls | Where-Object { $_.status -eq 'pending' })
    $reported = 0L
    $accounted = 0L
    foreach ($call in $calls) {
        $reported += [long]$call.reported_total_tokens
        $accounted += [math]::Max([long]$call.reserved_tokens, [long]$call.reported_total_tokens)
    }
    Write-Host ""
    Write-Host "[$stamp 北京时间] $Run | $(Get-StatusLabel $state.scene.status)"
    Write-Host "可用于装配任务：$done / $($jobs.Count)（可能仍有待验证项；不是总耗时百分比）"
    Write-Host "模型请求：$($calls.Count) 次，待返回或待确认：$($pending.Count) 次"
    Write-Host "预算记账：$accounted 词元；服务已报告：$reported 词元（两者不是同一口径）"
    Write-Host "场景装配记录：$(@($state.scene.rounds).Count) 轮；账本距上次更新：$age 秒"
    foreach ($job in $jobs) {
        $entry = $state.scene.jobs.($job.id)
        $label = if ($entry) { Get-StatusLabel $entry.status } else { '尚未完成' }
        if ($entry.dirty -and $entry.status -in @('candidate_ready', 'assembly_ready')) { $label = '已标记修复或依赖变化，需要更新' }
        Write-Host "  $($job.id)：$label"
    }
    if ($state.scene.pending) {
        $kind = switch ($state.scene.pending.kind) {
            'asset' { '资产构建、检查或审查' }
            'asset_review' { '重审现有资产，不重新建模' }
            'scene' { '完整场景装配与检查' }
            'plan' { '场景计划' }
            'repair' { '局部修复计划' }
            default { $state.scene.pending.kind }
        }
        Write-Host "当前未归档阶段：$kind，目录 $($state.scene.pending.directory)"
    }
    $last = $calls | Select-Object -Last 1
    if ($last) {
        $phase = switch -Regex ($last.phase) {
            '^asset_review:(.+)$' { "资产视觉审查：$($Matches[1])"; break }
            '^asset:(.+)$' { "资产生成或修复：$($Matches[1])"; break }
            '^scene_plan$' { '场景计划'; break }
            '^scene_review$' { '整场景视觉审查'; break }
            '^scene_repair$' { '场景修复计划'; break }
            default { $last.phase }
        }
        $status = switch ($last.status) {
            'pending' { '等待返回或需要确认' }
            'done' { '响应已保存' }
            'failed' { '请求失败' }
            default { $last.status }
        }
        Write-Host "最近请求：$phase；$status"
        if ($last.status -eq 'pending') {
            $seconds = [math]::Max(0, [int]($now.ToUnixTimeSeconds() - [double]$last.started_at))
            Write-Host "该请求登记后已过 $seconds 秒。账本不能单独证明进程存活或请求仍在服务端执行。"
        }
    }
    $resultPath = Join-Path $runRoot 'result.json'
    if ($state.scene.status -ne 'running' -and (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        Write-Host "结果文件：$resultPath"
    }
}

if ($Watch) { Write-Host '只读监视，每隔数秒刷新；Ctrl+C（中断快捷键）只关闭此监视窗口中的脚本。' }
do {
    try { Show-SceneSnapshot }
    catch {
        if (-not $Watch) { throw }
        Write-Warning "读取进度失败，下次刷新重试：$($_.Exception.Message)"
    }
    if ($Watch) { Start-Sleep -Seconds $IntervalSeconds }
} while ($Watch)
