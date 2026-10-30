# 移动双臂操作模块

这是 `robot_operation`（机器人操作包）。正式入口为 `run.ps1 operate`（操作子命令）。原有单次请求、执行和重放入口保持离线；新增联合改进入口默认离线，只有用户显式提供执行开关与固定实现摘要后才使用项目模型客户端。模块不下载依赖。开发者样例与模型输出必须分别标记。基准机器人公共配置由研究者更新；本次实现不改动 `selected50_protocols`（任务集）。

原首版测试见[历史离线验证记录](robot_operation_verification.md)，不能用它认证新源码。当前联合流程、命令和验证范围见[联合升级说明](joint_upgrade.md)及[两轮审查](joint_upgrade_review.md)。

## 当前能力与边界

- 从已发布场景派生独立动力学版本，将平面根替换为自由根；按名称和类型迁移关节，保留约束顺序，检查全部身体位姿，记录来源摘要。静置检查在副本进行，不修改任务初态。
- 操作图支持有限分支和循环；任务阶段、对象身份、初始与最终条件独立于图节点。节点完成不等于任务完成，任务契约通过不等于基准评分通过。
- 使用现有整机的三组转向/驱动轮，按实际圆柱几何读取轮半径；速度映射、执行器和关节限位以当前固定资源为准。其他轮组结构须增加明确适配，不能静默猜测。
- 采用 A*（启发式路径搜索）、RRT-Connect（双向快速扩展随机树）、阻尼最小二乘逆运动学；执行通过驱动器推进 MuJoCo（多关节动力学仿真器）。候选碰撞搜索只在副本改位姿。
- 支持位置伺服、双臂联合目标、抓持/支撑证据、轴向和圆弧操作、设备门控。接触负荷升高时减速，越界停止后通过显式失败分支退让，不是精确力控制；柔性物体来自生成资产，模块不把柔性试纸条替换为刚体。
- 抓取要求不同夹持身体上的相向接触，并保留相对位姿证据，不创建抓取焊接。首版针对双面夹持；吸附等其他抓取方式需要专门检测器。抓持阈值仅是工程代理指标，不代表真实硬件的力闭合认证。
- 未绑定检测器的机械事件会阻塞任务。检测器还必须观察到机器人或携带物与该事件接口部件的接触，模型软件请求不能直接触发机械事件。
- 轮式导航、图像采集、命令重放及异常路径由开发者合成场景测试。新增原厂整机开发者短链包含实际轮式停靠、开盖、抓持、搬运和释放；通过与否须查看当前验证收据，不能把转换兼容或部分阶段完成当作整条任务通过。
- NP-001 完整任务、复杂柔性接触、实际夹具配合、模型规划效果和全部 50 份任务均仍需正式实验。现有场景若静置或交互检查失败，会保留失败，不自动降低检查标准。

## 输入契约

所有 JSON（结构化数据格式）数值必须有限。四元数为 `[w,x,y,z]`，长度单位米，角度单位弧度。文件内容摘要绑定输入、任务、场景和图；场景改变后旧图失效。

任务契约 `task.json`（任务文件）字段：

| 字段 | 内容 |
|---|---|
| `version` | 当前为 1 |
| `input_sha256`、`public_profile_sha256` | 输入清单及公共环境摘要 |
| `provenance` | `developer_fixture`（开发者样例）或 `model_generated`（模型生成） |
| `bindings` | 任务对象身份到场景身体名称的映射 |
| `initial`、`final` | 初态和最终状态检测条件 |
| `stages` | 含 `id`（编号）、`source`（输入依据）、`depends_on`（依赖）和 `goal`（完成条件）的有序阶段 |
| `invariants` | 每步必须保持的检测条件，可省略 |
| `mechanical_detectors` | 每项含 `event`（资产已声明的机械事件编号）、`source`、`condition`（含正持续时间的条件） |
| `devices` | 每项含 `id`、`interface`（已声明软件接口）、`start_condition`（启动条件）、`running_condition`（运行条件）、`duration_seconds`（运行逻辑秒数） |
| `allow_logical_time` | 只有明确为真时允许设备等待的逻辑时间加速 |

检测条件均有 `kind`（类型），可选 `duration`（连续成立秒数，默认 0）：

| 类型 | 必要参数与含义 |
|---|---|
| `all` / `any` / `not` | 非空 `items`（子条件列表）或单个 `item`（子条件） |
| `joint_range` | `joint`（关节名）、`range`（闭区间） |
| `contact` | `body_a`、`body_b`（身体名）；可选 `min_force_n`（最小接触力） |
| `relative_pose` | `site_a`、`site_b`（坐标点）；可选 `offset`（在后者坐标系的偏移）、`position_tolerance`、`angle_tolerance`（位置、角度容差） |
| `body_stable` / `base_stopped` | 前者需 `body`；可选 `speed`（线/角速度工程阈值） |
| `held` | `arm`（手臂）、`body`、`tcp`（工具中心坐标点）；检查双面接触和相对位姿保持 |
| `supported` | `body`、`support`（支撑身体）；检查接触、释放与稳定 |
| `device_state` | `device`（设备编号）、`state`（空闲 `idle`、运行 `running`、完成 `complete`、故障 `fault`） |

操作图 `graph.json`（操作图文件）含 `version=1`、`scene_sha256`、`task_sha256`、`provenance`、`entry`（入口节点）、`nodes`（节点列表）。每个节点必须有 `id`、`stage`（任务阶段）、`operation`（操作类型）、`parameters`（参数）、`precondition`（前置条件）、`completion`（完成条件）、`timeout`（仿真秒）、`on_success` / `on_failure`（后继节点或终点 `success` / `failed`）。可选 `max_visits`（访问次数上限，默认 1）。

节点参数：

| 操作 | 参数 |
|---|---|
| `observe` / `check` | 无额外参数；以节点检测条件为准 |
| `wait` | 可选 `seconds`（仿真等待）、`logical_seconds`（设备逻辑等待） |
| `joint_move` | `arms`（手臂列表）、`positions`（机器人关节名到目标值）；可选 `include_lift`（使用共享升降） |
| `cartesian` / `dual_arm` | `targets`（目标列表），每项含 `arm`、`position`、`quaternion`，可选 `frame`（目标相对坐标点）。前者可选 `contact`（按接触路径执行）；后者必须两个目标 |
| `grasp` | `arm`、`body`、`controls`（夹爪执行器名到闭合上界）；可选 `approach`（同末端运动参数）、`closure_speed`（闭合速度）、`contact_force_n`（两侧各自最低接触力目标）。实际驱动逐步闭合，仍受穿透、总接触力和节点时间边界约束；不能把接触力目标视为精确力控制。 |
| `place` | 同抓取，另需 `support`；可选 `approach`、`retreat`（退离的末端运动参数） |
| `insert` / `press` | `arm`、归一化 `axis`（轴）、`distance`（有符号距离）；可选 `frame`、`hold_arms`（保持原位的另一臂） |
| `rotate` | `arm`、`axis`、`angle`（有符号角度）；可选 `frame`、世界坐标 `pivot`（旋转中心）、`hold_arms` |
| `navigate` | `transport`（收拢的关节运动参数）、`goal`（平面位置和偏航）、`bounds`（平面搜索区域的左下/右上坐标） |
| `dock` | `transport`、`bounds`、`arm`、`interface`（目标操作接口）、`reach_targets`（可达性目标）；可选 `radius`（候选距离） |
| `device` | `device`（任务设备编号）；启动条件由固定检测器确认 |

接触路径可用 `stop_when`（停止检测条件），仅允许已有固定检测器，限定行程耗尽仍未满足时返回失败。`cartesian`（末端运动）可显式设置 `include_lift`（参与升降），目标可指定 `interface`（实际接口局部坐标系）；共享升降影响另一臂，保持另一臂时应一并声明其目标。抓持稳定后锁定相对位姿参考，后续滑移不能通过重新读取检测器清除。

所有节点可声明 `allowed_contacts`（预期接触身体对）放在参数中；这不会绕过穿透和接触力检查。操作图无法访问文件系统、修改仿真器、调用任意代码或直接设置成功状态。

## 离线入口与交付

在 `D:\harness\new_harness` 中执行。所有命令的输出路径必须是不存在的新目录。以下变量由执行者替换，不能把示例摘要当真实摘要。

```powershell
# 查看已发布场景摘要，不运行仿真。
.\run.ps1 operate inspect --scene $sourceScene

# 派生操作场景并静置验证；会进行离线动力学检查。
.\run.ps1 operate prepare --scene $sourceScene --expected-sha $sourceSha --output $dynamicScene

# 准备模型请求，零联网调用；公共配置必须含 version、robot、workspace。
.\run.ps1 operate request --task-root D:\harness\selected50_protocols\NP-001 --scene $dynamicScene --expected-sha $dynamicSha --public-profile $publicProfile --output $requestDirectory

# 外部恢复模型接口后，将响应作为显式文件导入；响应包含 task、graph 和模型来源。
.\run.ps1 operate accept --request $requestFile --response $responseFile --output $acceptedDirectory

# 完整连续执行；试错可显式传 --mode trial。
.\run.ps1 operate run --scene $dynamicScene --expected-sha $dynamicSha --task $taskFile --graph $graphFile --output $runDirectory
.\run.ps1 operate replay --run $runDirectory

# 明确由用户启动：1次完整执行 + 3次重复 + 10次小扰动，合计14次。
.\run.ps1 operate repeat --scene $dynamicScene --expected-sha $dynamicSha --task $taskFile --graph $graphFile --output $repeatDirectory

# 输出公开工程默认值；可用 --settings 指定部分覆盖。
.\run.ps1 operate defaults
```

`accept`（接收响应）保存输入请求、原始响应及提供商、请求模型、响应模型；外部提供的来源字段会标明为外部响应，模块不声称已自行验证服务商。

执行输出有 `initial.json`（完整初态）、任务与操作图、`steps.jsonl.gz`（逐步控制和真值）、`events.jsonl`（阶段、失败和同步相机参数）、`images`（图像目录）、`recording.json`（版本与数据状态）、`result.json`（结果）、`files.json`（证据摘要）。逐步数据是流式写入，不在内存缓存整条轨迹。`--no-images` 只适用于调试，不能标为完整数据集。图像生成失败保留物理结果并标记数据不完整。

重放重新施加记录的控制输入并由相同检测器推进设备与连接，检查连续时间、关节位置/速度、完整积分状态及事件；不逐帧覆盖物体位姿。不同代码或仿真器版本会拒绝直接比较。运行目录内的 `implementation_sources`（固定源码副本）保存启动层实际绑定的工程源码，不含凭据。代码升级后可使用该目录中的启动脚本重放：`python -B -P <运行目录>/implementation_sources/runtime_bootstrap.py --module robot_operation -- replay --run <运行目录>`。长期归档还须保留所引用的完整场景和对应第三方依赖。

## 恢复、修复与预算

本地重搜最多 3 次，仅重搜路径/逆运动学失败；已发生的接触动作不自动倒带。实际失败通过图中的显式失败分支恢复。试错检查点保存物理状态、随机数、检测计时、设备、控制状态与图执行位置；正式连续执行禁止恢复检查点。

失败输出 `feedback_request.json`（待模型处理请求），不会触发联网。`operate revise`（导入修订）需要 `--previous-run`、`--graph` 和 `--output`，保持原任务契约，最多 2 次图修订；后续从任务初态重新执行。更换场景还需 `--new-scene`、`--new-scene-sha`、`--defect-report`，最多 3 次场景修复，旧图和旧轨迹不复用。

`operate diagnose --scene ... --expected-sha ... --output ...`（独立诊断）当前复核来源场景初态几何穿透，确认后输出现有场景生成入口可读的 `scene_feedback.json`（场景反馈）。其他失败仍可记录和请求诊断，但不能冒充该检查器已经确认的夹具、机构或布局缺陷。场景生成的修复继续使用已有流程；没有模型接口时请求保持待处理。

默认预算：每节点时间来自操作图、总仿真 1800 秒、最多 500 次节点执行、每节点最多 10 次访问；完整参数随记录保存。重复验证冻结这些参数，扰动只在初始化前施加，范围为底盘平面位置 ±5 毫米、偏航 ±0.5 度。不筛掉失败案例，不预设成功率。

## 开发测试与实验分工

```powershell
.\test.ps1 -p test_robot_operation.py
.\test.ps1 -p test_operation_contracts.py

# 可选的开发者合成演示，输出独立场景、图像和命令重放证据。
.\.venv\Scripts\python.exe -X utf8 -B -P .\src\runtime_bootstrap.py --script .\scripts\operation_demo.py -- --output .\.cache\operation_demo
```

这些是零模型调用的工程测试，不是 NP-001 正式实验。研发人员实现与测试代码；研究者更新公共机器人配置、确认场景与任务契约，恢复模型接口后亲自运行正式实验。正式实验仍需记录服务预算、场景版本和实际成功率。
