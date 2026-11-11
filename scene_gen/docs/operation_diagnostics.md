# 操作诊断、证据复用与资产返修

本次增强现有操作联合改进流程，保留资产生成、场景装配、固定任务条件和原执行器。诊断由通用量测工具组合而成，不为每种仪器或故障增加一个专用检查器。正式模型实验由用户启动。

## 入口与数据流

`operate improve`（操作联合改进）自动执行：记录初态、节点起态和失败现场 → 量测与轨迹证据 → 原因候选 → 补查或操作/资产/夹具返修 → 复查及完整初态连续执行。

工具清单：

```powershell
Set-Location D:\harness\new_harness
.\run.ps1 operate diagnose-tools
```

`diagnose-tools`（诊断工具目录）列出参数；状态量测提供 `state`（状态选择），可选 `initial`（已有执行的真实初态；未绑定执行时为场景初态）、`node_start`（最后执行节点的完整起态）、`failure`（真实失败现场）。整段轨迹、局部几何和证据详情不随状态标签重新量测。旧记录没有节点起态时返回缺失证据，不能推造状态。查询可附 `question`（要解决的问题）、`hypotheses`（候选解释），这两项不参与量测去重。

| 工具标识 | 能力与边界 |
|---|---|
| `contacts`（接触） | 接触双方、位置、深度及力；正常接触不授权夹具返修 |
| `configuration`（给定姿态） | 在副本中检查有界关节姿态的接触；可提供夹具设计假设，不证明该姿态可达，不把计算力当作实际执行力 |
| `relative_pose`（相对位姿） | 相对平移、转角，指定参考轴时计算轴向和横向偏差 |
| `joint_state`（关节状态） | 标量关节位置、速度、限位和关联执行器控制/力 |
| `geometry`（几何） | 实际碰撞几何参数和关联接口来源声明；不把外形尺寸当成内腔间隙，不把声明当成独立认证 |
| `distance`（距离） | 复用现有带一致性检查的几何量测；精确值、分离下界、未知分别保存 |
| `condition`（条件） | 读取既有操作图的固定前置或完成条件，不允许改判据来获取通过 |
| `mechanism`（机构） | 采样声明行程；锁止时不强行解锁，多关节依赖未认证或行程无规格依据时只给观测 |
| `fit`（配合） | 在副本中做给定起姿的直线几何试探；不代表实际执行成功，不直接授权改仪器 |
| `trace`（轨迹） | 流式分析录制数据，定位阈值候选、接触部件、条件失败及相对滑移；保留有界前后窗口 |
| `probe`（对照试探） | 同一完整状态、相同随机状态下，分别通过原执行器执行已有动作及单参数变体 |
| `evidence_detail`（证据详情） | 按证据编号、字段路径和分页读取完整证据；不重新仿真，不授予返修权限 |

对照只能修改动作已经使用的目标、关节位置、夹爪控制、轴、距离、转角或导航目标；不能改对象身份、前置/完成条件、接触许可、仪器规格或引擎配置。目标修改保留手臂、接口及参考系。没有满足前提的完整状态时记录未知，不拿错误起姿的失败证明资产有缺陷。

## 离线补查已有执行

输入是一个执行目录，里面包含原有录制、文件摘要、任务和操作图。输出必须是独立新目录，不写入原运行或场景。下面的参数由用户选择实际运行路径，不填写虚构摘要。

```powershell
$execution = Read-Host '输入已有执行目录的绝对路径'
$queryFile = Read-Host '输入诊断查询文件的绝对路径'
$diagnosticOutput = Read-Host '输入新的诊断输出目录，须在 D 盘'
.\run.ps1 operate diagnose-query --run $execution --query $queryFile --output $diagnosticOutput
# 同一输入与实现下继续；重复量测直接复用。
.\run.ps1 operate diagnose-query --run $execution --query $queryFile --output $diagnosticOutput --resume
```

`diagnose-query`（离线证据补查）不调用模型。查询文件示例采用 JSON（结构化数据格式）；其中对象和节点名必须来自实际场景及操作图：

```json
[
  {"kind":"contacts","state":"failure","question":"首次受阻涉及哪些部件"},
  {"kind":"trace","state":"failure","window":100},
  {"kind":"joint_state","state":"failure","joints":["robot__left_x"]}
]
```

最后一条仅适用于包含该关节的开发者样例。正式机器人应使用工具目录和模型内的真实编号。

产物：`report.json`（诊断报告）、`ledger.json`（持久证据账本）、`configuration.json`（输入及实现绑定）。`trials/evidence`（完整证据归档）保存量测原文及摘要绑定；长轨迹窗口在账本中仅预览首尾，完整窗口可通过证据详情读取。有对照时，试探目录另保存原动作和变体的控制轨迹、结果和可重放资料。旧 `diagnose-run`（原有固定诊断）入口继续保留，不自动替换历史记录。

## 证据有效性和预算

量测缓存绑定场景摘要、状态摘要、工程设置、工具版本；依赖执行记录的工具还绑定原运行回执。标准化默认参数、对象列表、数值及字段顺序后再判等，不使用自然语言相似度推断等价。已覆盖的等距采样、对象子集和更小轨迹窗口可复用已有更完整的量测。

`valid`（有效观测）、`unknown`（无法确定）、`invalid`（检查不成立/能力缺失）、`interrupted`（中断）分开保存。有效观测不等于动作成功或根因已确定。工具失败不会授权返修；无效及中断请求也保留缓存和预算，防止同样请求不断重跑。

信息增益依据实际观测与覆盖范围的变化，不接受模型自评。只换问题措辞、输出路径或证据编号不能增加信息。缓存复用不消耗新的量测预算；连续没有增加有效信息的补查会进入明确待处理状态。

以下是可配置的公开工程默认值，均位于操作 `Settings`（工程设置），通过原有 `--settings`（设置文件）提供；不是厂商规格或论文结论：

| 字段 | 默认值 |
|---|---:|
| `diagnostic_queries`（新量测总数） | 256，包含自动证据、无效请求和中断预留 |
| `diagnostic_batch`（每批查询数） | 32 |
| `diagnostic_stalls`（连续无增益批次） | 3 |
| `diagnostic_probe_seconds`（单次动作试探上限） | 5 秒；一次原动作/变体对照最多预留 10 秒 |
| `diagnostic_total_sim_seconds`（累计试探预留） | 120 秒 |
| `diagnostic_trace_steps`（单次轨迹扫描上限） | 1,000,000 步 |
| `diagnostic_trace_window`（异常前后窗口各自上限） | 200 步 |

完整场景摘要变化会使接触、位姿、机构采样和轨迹等既有量测失效。仅局部几何查询可在其全部已返回参数、接口声明、网格顶点与面、工程设置和工具版本均未变时跨场景复用；新记录绑定新场景并引用原证据，不增加量测消耗。这不能推导该对象在新环境中仍然可达或无碰撞。归档缺失或被修改时拒绝复用。新记录含固定条件的失败/保持量测，旧记录未保存的滑移量仍是未知。

## 返修分流与复查

联合协调器允许操作修订、仪器资产修复、夹具设计试验、场景修复、继续补查或明确待处理。候选原因允许多个并存；局部对照改善只支持对应操作参数的局部效果，不证明唯一根因。

仪器资产修复需要固定检查授权；夹具可凭具体干涉或真实保持失败进入设计试验。夹具分类来自实际场景任务角色及末端绑定。普通支撑接触、单纯搜索失败、工具错误和过期证据不能作为夹具返修依据。仿真不稳定与执行器程序故障进入工程问题分支。

新返修请求必须包含 `reason`（原因）、`instances`（待修改实例）、`evidence_ids`（实际证据编号）、`checks`（初态复查查询）。修改前实际执行这些检查，确认检查方法有效且覆盖修改对象。若引用初态的接触、给定姿态或机构/配合采样，复查必须覆盖原量测的对象、姿态与采样路径，不能用读取几何参数替代。修改后重新量测并比较原干涉、新干涉、覆盖和阶段完成情况。未解决原问题不能进入最终成功确认；新缺陷或覆盖退化不能覆盖基线。末端/夹具设计的反馈只送给所选实例对应的原生成任务。

资产规格、原厂机器人参数、固定任务和成功判据继续由已有生成/操作契约约束。正式最终执行仍从完整初态连续运行，不拼接局部试探的成功片段。局部试探独立标记，不能成为正式成功轨迹。

本版复查使用初态量测和完整连续任务执行；尚不支持把旧版本失败状态自动迁移到任意新拓扑模型，也不假装已认证全部多关节依赖、未知材料属性或柔性形变。需要的新物理量通过注册通用工具扩展；模型不能在运行时修改检查器代码或阈值。

## 工程验收与正式运行

离线工程检查由助手执行，禁止外网连接；开发者预置响应与真实模型来源分开记录。

```powershell
.\.venv\Scripts\python.exe -B scripts/test_offline.py test_operation_diagnostics test_joint_upgrade test_joint_failure_probes test_joint_upgrade_boundaries test_robot_operation test_operation_contracts --receipt .cache/operation_diagnostics_user_check.json
```

收据路径须未存在。本次交付检查记录位于 `.cache/diagnostic_upgrade_20261028`，以最终交付收据为准。

收尾前执行过全套 1,028 项离线检查，1,023 项通过，5 项失败在隔离的改动前源码副本中同样复现：两个场景修复样例缺少进度接口、两个场景修复状态断言不符、一个文件硬链接权限错误。原始日志和改动前复现记录均保留，未修改这些样例或掩盖失败。最终收尾代码以 `final_affected_checks.json`（最终受影响流程检查收据）及 `delivery_receipt.json`（交付收据）为准；全套检查没有在收尾改动后再次运行。

用户正式运行所需输入：已发布场景、对应原场景生成运行、原资产资料包、白名单任务输入目录、公共机器人配置、用户已有提供商配置。必须新建联合运行，不能用新代码原地恢复绑定旧实现的实验。

```powershell
$sourceScene = Read-Host '已发布场景目录'
$sourceRun = Read-Host '对应原场景生成运行目录'
$assetPackage = Read-Host '原资产资料包目录'
$taskInputs = Read-Host '含 input_manifest.json（输入清单）的任务输入目录'
$publicProfile = Read-Host '公共机器人配置文件'
$providerConfig = Read-Host '已有提供商配置文件，不输入密钥'
$jointRun = Read-Host '新的联合运行输出目录，须在 D 盘'
$receipt = Get-Content .cache/diagnostic_upgrade_20261028/delivery_receipt.json -Raw | ConvertFrom-Json
$sceneInfo = .\run.ps1 operate inspect --scene $sourceScene | ConvertFrom-Json
# 默认离线，生成请求并等待；由用户显式追加执行参数才调用正式模型。
.\run.ps1 operate improve --run $jointRun --scene $sourceScene --expected-sha $sceneInfo.scene_sha256 --task-root $taskInputs --public-profile $publicProfile --seed-run $sourceRun --package $assetPackage --provider-config $providerConfig
# 新建另一输出目录时，在同一命令末尾增加：
# --execute --expected-implementation $receipt.implementation
```

如来源使用额外规程文件或特定生成配置，应沿用来源参数。共享模型预算仍为每规程 800 次调用、240,000,000 预留令牌、80 轮候选执行、每资产 60 次候选构建；来源账本消耗继承。诊断试探预算不替代这些上限，也不是预计实际费用。

联合输出包含 `state.json`（状态及诊断账本）、`progress.json`（进度与诊断消耗）、`diagnoses`（诊断报告）、`diagnostic_trials`（局部对照）、`executions`（完整候选试执行）、`final`（最终连续执行）及原资产/场景生成档案。模型自主归因质量、夹具设计收敛率和完整规程成功率由用户后续正式实验评估。
