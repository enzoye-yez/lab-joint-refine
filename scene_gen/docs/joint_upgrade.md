# 资产与操作联合改进

本次升级沿用资产作者、场景装配、固定检查器和机器人操作模块；新增协调入口，不替换原框架。正式模型接口和 NP-001、NP-002 实验没有启动。开发者预置响应、参数化工具与模型生成来源分别记录。

## 流程与边界

独立任务条件 → 冻结条件 → 操作图 → 真实驱动试执行 → 独立诊断 → 操作修订或场景/资产修复 → 重新装配、接口重绑定 → 新图 → 完整初态连续验证。

- `operate improve`（联合改进）默认离线，导出实际可发送的预算内请求及引用资料包；不会创建在线客户端或读取密钥。只有显式 `--execute`（执行模型调用）才使用项目现有客户端。
- 任务编译不读取候选图。任务文件冻结后只允许有明确语义映射的引用重绑定，不能修改目标或检测标准。无法核实的要求保留未知。
- 操作图失败默认进入诊断和路由。搜索未找到解不是不可达证明，也不授权修改资产。机构自身干涉、来源初态穿透等独立量测可支持几何修复；配合直线试探只给观察，仍需判断起姿是否正确。
- 当前自动几何修复需要 `--seed-run`（来源场景运行）和原 `--package`（资产资料包）。单独的模型文件可完成操作修订，但不能虚构原作者工作区。
- 来源运行只读复制；子流程共享父账本。新配置、新源码或新输入不能在原联合运行上偷偷迁移。
- 末端目标支持真实接口局部坐标；停靠搜索可联动左右臂与升降。实际执行不靠修改物体位姿完成动作，不创建隐形抓取焊接。
- 新派生的动力学场景显式启用凸几何多点接触，避免将夹指接触面压成单个角部点；在转换记录中保存来源和派生的求解开关。原厂资源、几何、摩擦值与检查阈值不因此改写，旧场景也不原地迁移。
- 抓取可声明有界的两侧接触力目标，逐步闭合实际夹爪；取得稳定夹持后锁定参考，滑移不能被重复读取抹除。这不是精确力控或通用力闭合证明。
- 接触路径支持可选 `stop_when`（真实检测停止条件）：只在限定行程内观察固定检测器，条件未达到则失败。释放后的承托仍独立验证。

## 小型构建工具

`asset_generation.construction.Builder`（参数化构建器）提供四个组件，采用米、弧度以及 `compiler angle="radian"`（弧度编译设置）：

| 方法 | 参数与输出 |
|---|---|
| `hinge`（铰链） | 父节点、名称、原点、单位轴、含零位的限位；销、套和关节共用参数 |
| `slide`（滑轨） | 父节点、名称、原点、单位轴、限位、轨长、宽度、间隙；要求最大位移加半滑块宽小于半轨长 |
| `container`（空心容器） | 外形三尺寸与壁厚；五面碰撞几何，保留开口 |
| `bracket`（安装支架） | 板三尺寸及矩形通孔两尺寸；真实孔和安装坐标点 |

`attach`（特征对齐）根据两组右手正交坐标系计算刚体安装变换。它不做自动配合认证或一般 CAD（计算机辅助设计）求解。组件返回可继续组合的 MJCF（MuJoCo 模型描述格式）节点。生成者需把构建器的 `records`（记录）放入接口清单 `construction_records`（构建记录）；来源为 `engineering_constructor`（工程构建工具），`certified=false`（未经质量认证）。检查器仍独立检查生成模型。

可选接口 `operation`（操作信息）包含 `status`（声明或未知）、`basis`（逐字段依据），以及适用的轴、关节、行程、间隙、抓取几何和配合接口。依据类型仅为来源规格、量测或假设。实际关节、部件、坐标点与几何引用须存在且关联正确；声明本身不算测量。旧接口无需补造这些字段，缺少时诊断返回未知。语义编号变化使用 `replaces`（旧编号映射），拒绝歧义映射、改变资产身份和缩短已声明行程。

## 预算、恢复与证据

每规程共享 800 次调用、240000000 预留令牌、80 轮进入执行的候选方案、每资产 60 次候选构建。来源账本和可确认的候选消耗一并继承。新联合流程不使用旧独立修订入口的 2 次图修订/3 次场景修复上限；旧命令为兼容归档保留原语义。

模型调用、离线导入的真实模型响应、分页批次均记账；开发者样例导入不伪装为模型调用。未知请求保留全部预留、不自动重发。重复导入不能重置消耗。离线外部调用的来源字段是外部声明，不声称框架已向服务商核实。

本地搜索、节点访问和仿真仍有有限预算；局部补查每次最多 32 项、每项 2～56 个采样姿态、每联合运行最多 256 次不同补查。连续路径、力控稳定性和未采样姿态不会因离散几何检查通过而被认证。

候选比较保留具名缺陷、覆盖缺口、阶段完成情况和静态检查；部分改善可采纳，新增缺陷或覆盖退化留在探索分支。回退恢复对应资产/场景选择、准入结果和预览，保留全部账本、阶段及失败记录。相同输入和证据下重复失败机制受到限制；输入或证据实质变化后可再尝试。

每个试次都是明确标记的试错；最终验证从完整初态开始。正式连续记录禁止恢复检查点。中断没有完整回执时不自动重新执行。每步检查暂停，执行边界校验源码与依赖，变更后的结果不能采纳。

`progress.json`（进度）和 `scene_document.md`（场景文档）分别记录完整性、静态质量、动力学准入、操作结果及证据缺口。质量失败不等于场景不存在；任务通过不代表真实仪器复现精度或正式规程成功。

## 命令

在 `D:\harness\new_harness` 执行；先给变量赋实际路径，不复制示例占位内容当作摘要。

```powershell
# 零模型调用：先读取场景摘要。
$sceneInfo = .\run.ps1 operate inspect --scene $sourceScene | ConvertFrom-Json
.\run.ps1 operate improve --run $jointRun --scene $sourceScene --expected-sha $sceneInfo.scene_sha256 --task-root $taskInputs --public-profile $publicProfile --seed-run $sourceRun --package $assetPackage

# 导入当前请求的响应；然后继续离线推进，下一请求会再次等待。
.\run.ps1 operate improve-import --run $jointRun --response $responseFile
.\run.ps1 operate improve --run $jointRun --resume
.\run.ps1 operate improve-page --run $jointRun --request $requestIndex --query $pageQuery
.\run.ps1 operate improve-status --run $jointRun
.\run.ps1 operate improve-pause --run $jointRun
.\run.ps1 operate improve --run $jointRun --resume --continue-paused

# 只调整关节查看结构，不推进动力学、不生成任务通过证据。
.\run.ps1 operate view --scene $dynamicScene --expected-sha $dynamicSha

# 独立诊断和控制输入重放，不调用模型。
.\run.ps1 operate diagnose-run --run $executionDirectory
.\run.ps1 operate replay --run $executionDirectory
```

响应封装包含当前 `request_id`（请求编号）、完整 `delivered_batches`（已处理批次序号）、`provenance`（来源）、`value`（当前阶段响应）。真实模型来源还需 `model_provenance`（提供商、请求模型、实际响应模型）；必须与运行选择一致。开发样例只允许新建时显式 `--developer-fixture`（开发者样例），禁止在这种运行上启用在线调用。

请求索引列出每批实际消息、预算、内容摘要和资料根。分页查询含 `reference`（已授权引用）、`path`（字段路径）、`offset`（起点）、`limit`（长度）。图片按批保留，不靠删图降低预算。

后续余额恢复后，由用户显式启动正式运行：在上面的新建命令提供实际模型配置 `--provider-config`（提供商配置），并增加 `--execute --expected-implementation $verifiedImplementation`（与新测试收据一致的源码摘要）。恢复在线运行也必须明确传这两个参数。不要把开发者录像的任务或预置响应当成正式实验输入。

## 验证和未验证范围

复现开发者短链：

```powershell
.\.venv\Scripts\python.exe -B scripts/verify_joint_upgrade.py --output .cache/joint_acceptance_new --developer-strategy tests/fixtures/ffw_joint_strategy.json
.\.venv\Scripts\python.exe -B scripts/test_offline.py
```

脚本禁止外部网络，使用原厂固定版本 FFW-SG2（双臂移动机器人）和显式开发者夹具。冻结任务后读取有来源摘要的固定开发者策略，注入空抓，导入预置修订，再从完整初态执行与重放。生成三视角录像、任务/图、失败现场和实际控制记录；是否通过以本次 `verification.json`（验证收据）为准。脚本可显式复用有来源摘要的开发者策略，不能据此声称模型自主修复。

本次没有证明完整 NP-001/NP-002 操作成功率、模型自动收敛率、通用力闭合、真实硬件安全或一般轮组适配。复杂配合原因不明确时仍需补证；当前没有将任意几何假设直接送去自动重构。隔离构建后端沿用现有实现，本次不扩大执行权限。

上述复现命令使用明确标注来源的固定响应；脚本省略策略参数时的交互示教路径仅用于开发调试，重新测得的姿态可能导致不同接触结果，不在本次通过声明内。

## 本次固定版本交付

2026-10-25：正式接入后全量离线测试 **961 项通过，0 失败、0 错误、0 跳过**；测试前后源码、测试及脚本摘要一致。原厂整机短链在固定开发者响应下完成停靠、开盖、抓取、搬运和释放承托，随后从完整初态再次通过，30,010 步控制重放最大状态误差为 0。注入空抓确实发生在抓取节点，预置修订后重新执行，没有跳过前置失败。原厂清单 33 个资源摘要一致。

- 总交付收据：`.cache/joint_upgrade_20261025/delivery_receipt.json`。
- 全量测试：`.cache/joint_upgrade_20261025/full_offline_final_receipt.json`。
- 短链收据与录像：`.cache/joint_upgrade_20261025/acceptance_release/verification.json`、同目录 `short_chain.gif`（三视角动图）。
- 来源策略：`tests/fixtures/ffw_joint_strategy.json`，旁边保存来源与文件摘要。

查看本次实际结果：

```powershell
Set-Location D:\harness\new_harness
$joint = 'D:\harness\new_harness\.cache\joint_upgrade_20261025\acceptance_release\joint'
$progress = Get-Content "$joint/progress.json" -Raw | ConvertFrom-Json
.\run.ps1 operate view --scene $progress.scene --expected-sha $progress.scene_sha256
.\run.ps1 operate diagnose-run --run "$joint/executions/0000"
.\run.ps1 operate replay --run "$joint/final/1"
```

进度摘要在阶段边界刷新；执行中的实时轨迹写入执行目录，精确持久预算保存在联合运行的 `state.json`（状态账本）中。任务条件通过与静态质量、完整规程评分和真实硬件适用性分别报告。本次原厂短链不代表全部注入故障都已经在原厂整机上完成整链恢复；其他故障由具体单项测试和跨模块构建回归覆盖。完整 NP-001、NP-002 与正式模型自主修复仍待实验。
