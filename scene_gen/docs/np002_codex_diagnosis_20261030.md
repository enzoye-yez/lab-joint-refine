# NP-002 agent 诊断与修改意见（2026-10-30）

本文基于三份证据：用户提供的历史结果 `np002_repair02_scene_outputs_20261029`（671 次模型调用、47 轮场景修复后暂停），本轮在 devbox 上用 codex 登录态（`gpt-6.1-sol`）重新跑的 A15 单资产和 NP-002 小预算批次，以及对 `src/` 的通读。所有统计可用 `scripts/analyze_scene_quality.py` 复现。本文只提修改意见，没有改 agent 策略代码。

## 1. 本轮做了什么

| 项 | 内容 |
|---|---|
| 环境 | `~/np002_agent`（git 管理，首个提交为原始快照）；Python 3.11 `.venv`，依赖按原版本固定；MuJoCo 3.3.7 |
| 渲染 | Mesa 软件 EGL（llvmpipe，纯 CPU），所有 GPU 对进程隐藏；`scripts/env.sh` 统一设置 |
| 模型 | 新增 `src/asset_generation/codex_transport.py`：每次请求运行一次 `codex exec`（`gpt-6.1-sol`、只读沙箱、不持久化会话），返回 chat completions 形状，账本/缓存/预算/恢复不变；默认配置改为 codex，不读 API key |
| 入口 | `run.sh`、`test.sh`、`scripts/run-scene-batch.sh` 对应原 PowerShell 入口 |
| 测试 | 全量 1079 项；原代码在 Linux 上已有 27 项失败（多为 Windows 专属），本轮改动不新增失败 |

## 2. 证据

### 2.1 历史结果 repair02（671 次调用后暂停）

| 指标 | 数值 |
|---|---|
| 场景检查行 | 444 行：pass 342 / fail 50 / partial 49 / skipped 3 |
| 失败来源 | 50 项失败中 49 项是 `material_integrity`（资产内部材料完整性），1 项场景碰撞 |
| 材料完整性实例 | 111 个实例：pass 16 / fail 49 / partial 46 |
| 碰撞对 | 2134 对；穿透中位数 1.0 mm，50% ≤ 1 mm，71% ≤ 3 mm（容差 0.5 mm） |
| 不完整原因 | 46 个 partial 绝大多数是网格距离 `measurement_error`（"无法核对最近点、接触法向与实际碰撞形状支撑面"） |
| 最差实例 | 测序仪 774 对 / 大离心机 586 对 / cBot 407 对 / 超声仪 168 对 |
| 修复轮 | 47 轮：adopted 6、baseline_preserved 40；结局 coverage_lost 20、regressed 12、ineffective 8、improved 3、improved_partial 3 |
| 修复目标 | 每轮前 3 个目标中采样器出现 41 次、天平 30 次；测序仪、大离心机、cBot 从未进入 |
| 解决的检查 | 全程只有 3 项（cabinet、cutter 材料检查和 cold_rack 承托） |
| 资产版本 | 场景使用的 24 个任务中 20 个仍是首稿（早于第一次场景修复），只有 asset01–04 被改过 |
| 视觉审查 | 24 个任务全部 `unassessed`/`review_pending`，没有任何资产完成资产级视觉审查 |
| 令牌账 | 预留记账 2.16 亿（上限 2.4 亿），服务实际报告 4421 万：估算约为实际的 4.9 倍 |

**"小球"的来源。** 编译模型中没有占位球体。查看器默认显示 site 组 0–2，场景有 2320 个 site，其中 1822 个可见，都是资产声明的操作接口点（抓取、按压、插入等），光三个枪头盒就有 882 个。只有 12 个离自身几何超过 1 cm，是真正悬空的接口，例如安全柜 `filter_holder_entry` 偏 100 mm、测序仪 `waste_bottle_remove_site` 偏 46 mm、过滤装置 `pour_opening_site` 偏 42 mm。其余小球只是接口可视化，不代表问题位置；真正遗留的问题在检查报告里，查看器中并不可见。

**视觉上看不出的"缺陷"。** 采样器、离心机的 CPU 渲染外观合理；它们的失败项是按钮嵌入面板 1 mm、转子轴毂压入电机座 1 mm、卡扣舌嵌入卡座 9 mm、脚与机身之间留缝等毫米级内部关系。其中卡扣、铰链、紧固件本来就是设计上互相咬合的结构。

### 2.2 本轮新跑（codex / gpt-6.1-sol）

- **A15 磁力架单资产**：3 次调用。第 1 版因"机械事件须引用真实操作接口"失败；第 2 版材料完整性失败；第 3 版所有几何检查通过，渲染像样，但唯一剩下的 `release_probe` 是 partial：模型正确把管架建成可从磁座取下的可拆连接，框架却没有对应的释放探测，于是状态停在 `needs_check_support` 并直接退出。没有视觉审查，也不交给模型修，而这个问题模型改代码也改不掉。
- **NP-002 小预算批次**：场景规划阶段用了 17 次调用。原因是上下文投影按 32 KB 硬编码分页（`request_context.py:364,480`），模型先 5 次只写笔记、再 11 次 `context_read` 才交出计划；对 gpt-6.1-sol 这样的长上下文模型，这是纯粹的预算浪费。计划给出 25 个任务、124 个实例、3 张桌子。复杂资产的作者调用单次可超过 600 秒（asset01 首次超时，重试成功）。批次最终统计见第 7 节。

## 3. 根因：两个问题分别卡在哪里

### 问题 1：反馈不足以驱动修改

数值反馈本身并不缺信息：每条碰撞都有两个几何名、穿透深度、最近点和法向。真正缺的是三样东西：

1. **意图分类**：报告不区分"设计上应当咬合或嵌入"（卡扣、铰链轴、按钮入面板、紧固件）和"真正的穿模"。模型无从判断该消除哪一条，于是消除咬合就破坏连接，表现为 12 轮 regressed，每轮都新增 `disconnected_components`。
2. **可执行的修改方向**：报告给的是世界坐标点对，模型改的是 `model.py` 里的参数化代码。从"geom A 与 geom B 沿 -y 穿透 9 mm"到"脚本里哪个常量改多少"，要模型在脑中反推坐标变换。repair 请求只能交唯一文本替换补丁，模型看不到改后的结果。
3. **本地验证手段**：每一次尝试都要走一整轮"装配全场景 → 场景检查 → 比较"，一轮几十次调用（47 轮共 671 次调用），模型没有机会在资产内部快速试错。

另外，视觉审查结论 `findings` 是自由文本数组（`asset_generation/prompts.py:61-69`），没有部件名、方向、严重度字段，进入修复时也无法与检查行对齐。

### 问题 2：审查越来越严，通过的资产寥寥无几

主因不是视觉审查严，而是**数值门槛与比较规则**：

1. **零容忍的材料完整性**：容差 0.5 mm，对 6 万几何体的整场景逐对计数；同一资产的每个实例重复计数（5 支移液器、3 个管子各报一次）。111 个实例里只有 16 个能过。
2. **不可量测等于不通过**：网格距离核对失败记为 partial，单资产流程直接 `needs_check_support` 退出（`asset_generation/pipeline.py:175`）；场景修复比较时，只要修完后某检查从 fail 变成 partial，就判 `coverage_lost` 并回退。47 轮里有 20 轮因此作废，其中可能包含真实改善。
3. **调度饿死**：修复任务按报告顺序排，每轮只取前 3 个任务（`pipeline.py:718`），报告顺序又跟随计划顺序，所以采样器和天平被修了几十次，最严重的测序仪、离心机、cBot 从未被修。
4. **审查从未走到**：静态 scene-first 流程里资产只做"能否编译"的准入，质量检查推迟到诊断场景保存之后；修复一直没有收敛，所以 24 个资产没有一个完成视觉审查。即使审查标准本身合理，也没有资产能走到"accept"。
5. **能力缺口变成硬阻断**：A15 的可拆管架、复杂网格、卡扣机构这类框架尚不能探测的机制，一律记 partial，而 partial 被当作未通过。

## 4. 修改意见（按优先级）

### P0：让门槛可通过、修复能收敛

1. **材料完整性分级**（`asset_generation/material.py`、`check_report.py`）
   - 对声明为 latch / hinge / mount / 按钮的接口对，以及同一刚体内的几何，嵌入不超过设计上限（如 3 mm，或按部件尺寸取比例）时记 `info`，不计入 fail。
   - 只把跨实例穿模、可见穿透超过阈值、以及与已声明接口冲突的嵌入记为 fail；每个资产只上报按深度排序的前若干条，并去掉同资产多实例的重复。
   - 断开构件只在该 body 没有任何焊接/关节/接触见证时才算缺陷。
2. **"无法量测"不阻断**（`recovery.py`、`progress.py` 中的比较规则，单资产 `pipeline.py:175`）
   - `measurement_error` 和缺少探测器的机制记为 `unverified`，交给视觉审查兜底，并列入交付说明。
   - 修复比较按"确认缺陷数的净变化"判断：fail → partial 且没有新增确认缺陷，应算改善或至少中性，不再按 coverage_lost 回退。
3. **修复调度轮转**（`pipeline.py:685-719`、`repair_schedule.py`）
   - 按缺陷严重度（最大穿透 × 条数，或审查 blocking）排序，并加入老化（等待轮数越多优先级越高），替代按报告顺序取前 3。
   - 同一资产连续两轮 regressed 或 coverage_lost 就暂挂，把名额让给其他资产。
4. **资产级视觉审查前置**：静态模式下，每个资产通过编译准入后先做一次资产级审查（单图、单资产，成本低），把明显缺件或形状错误在场景装配前解决；场景阶段只处理跨实例问题。

### P1：把反馈变成可执行的修复单

5. **结构化 repair ticket**（`progress.repair_tasks` 已有雏形）：每条缺陷附带
   - 所属 body/geom，以及在**资产局部坐标系**下的最近点、法向和"沿法向移动 X mm 可消除"的建议位移；检查器已经有 normal 和 points，直接换算即可；
   - 是否涉及声明接口，意图分类（设计咬合 / 穿模 / 悬空）；
   - 该 geom 在 `model.py` 中的生成位置（Builder 已记录 `construction_records`，可据此回溯参数名）。
6. **审查输出结构化**：`findings` 改为 `{part, problem, evidence_view, suggested_change, severity}`；severity 只允许 blocking / should_fix / cosmetic，只有 blocking 阻止通过。
7. **局部高亮渲染**：把问题几何高亮后渲染一张资产局部图交给作者（`asset_generation/observation.py` 已有定向视图能力），补足模型看不到改动效果的问题。

### P1：作者改为编码 agent 循环（见第 5 节）

### P2：效率与工程问题

8. **上下文分页**：把 32 KB 页大小和 `review_context_bytes`（120 KB）改为配置项，对长上下文模型直接整包发送；NP-002 规划可从 17 次调用降到 1–2 次。
9. **令牌估算**：图片固定 16384、文本按字节计，导致记账约为实际的 5 倍，提前耗尽预算；对报告了真实用量的服务，应按实际用量结算，预留只用于发送前的上限判断。
10. **超时**：codex 下复杂资产作者调用超过 600 秒，建议 codex 配置用 1800 秒。
11. **已确认的工程 bug**（通读代码得出，第 6 节逐条列出）。

## 5. 是否引入 pi / codex 这类编码 agent

**结论：值得引入，但只用在单资产作者这一环；规划、审查和最终裁决保持现有的单次结构化调用。**

现在的作者是"单发"模式：首轮输出完整脚本，之后只能交唯一文本替换补丁，看不到构建错误、渲染和检查结果，每次试错都要经过框架的一整轮。repair02 的失败大多是几何记账类错误（嵌入深度、接缝、悬空接口、改一处断一处），这正是编码 agent 在本地快速迭代最擅长的事。

建议的形态：

- 每个资产一个工作目录，内含资料片段、参考图、当前 `model.py`，以及框架提供的命令行工具：
  - `check_asset`：在本地后端运行 `build` 加 `validate_artifact`，输出精简的结构化检查结果；
  - `render_asset`：输出多视角及问题高亮图；
  - `measure`：查询几何对距离或包络。
- 编码 agent 在 workspace-write 沙箱里只能改这个目录，一次会话内多次运行检查、看图、改代码，直到本地检查通过或用完步数，再提交 `model.py`。
- 框架照旧独立重建、检查和视觉审查，编码 agent 的自测结论不作为通过依据，可审计性不变。
- 账本把一次 agent 会话记为一次逻辑请求，保存最终 `model.py`、会话日志摘要和工具调用次数。

实现选择：

- **首选 codex exec**：本机已登录，本轮已验证能稳定调用 gpt-6.1-sol，自带沙箱（只读 / workspace-write）和 `--output-schema`，接入成本最低。
- **pi（开源轻量编码 agent）**：适合需要自定义工具协议、切换多家模型或深度定制循环的场景。它能否直接复用 ChatGPT/codex 登录态，我没有在这台机器上验证；若只能用 API key，就与"不消耗外部 API"的约束冲突。建议先用 codex 验证收益，再决定是否换成 pi 以获得更强的可定制性。

风险：单次会话耗时变长、调用次数难以预估（需要给会话设置步数和时间上限）；模型可能"为过检查而改检查"，必须保证工具只读框架的检查代码、不能改验收配置，提交物仍限定为 `model.py`。

最小实验：选 repair02 中最差的 3 个资产（测序仪 A19、大离心机 A06、cBot A18）以及 A15，比较"现有补丁循环"与"codex 编码 agent 循环"在相同调用预算下：材料完整性确认缺陷数、通过的资产数、退步次数、耗时。

## 6. 已确认的工程问题（代码通读）

| 位置 | 问题 |
|---|---|
| `scene_generation/pipeline.py:718` | 修复范围按报告顺序取前 `max_repairs_per_round` 个任务，导致调度饿死（见 3.2） |
| `asset_generation/pipeline.py:175` | `needs_check_support` 直接 break，不审查、不修复 |
| `asset_generation/model.py:267` + `provider_diagnostics.py:74` | HTTP 5xx 和发送后超时被标为结果不明，现有重试分支对它们不生效，批次直接暂停（codex 传输已改为可重试） |
| `asset_generation/contracts.py:119` | 作者 `deferred` 抛 `ContextLimitError`，恢复时命中同一缓存再次暂挂，没有出路 |
| `scene_generation/pipeline.py:817` | 完整模式不记录修复基线，退步修复不会回退 |
| `scene_generation/pipeline.py:111` | 每个 progress 事件都调用 `result()`，对大场景重新求摘要并写 5 个文件 |
| `scene_generation/layout_planning.py:30` | 布局细化传入空包围盒，没有用上真实几何 |
| `scene_generation/evidence.py:537,593` | 同一轮里观察失败被计数两次 |
| `asset_generation/probes.py:17,54` | 多构型资产的交互探测固定读 `model.xml` |
| `containers/asset-worker.Dockerfile` | 缺 scipy，容器后端的场景检查会 ImportError（本地后端不受影响） |
| `scene_generation/pipeline.py:173-197` | 场景计划校验失败的原因只保存在内存里，不落账本，事后无法复查 |

## 7. 本轮小预算批次结果

批次 `np002_codex_small_01`（150 次调用、8 轮场景修复、每资产 6 次）运行中；完成后用以下命令补充统计：

```bash
python3 -I scripts/analyze_scene_quality.py runs/np002_codex_small_01_np002
```

## 8. 复现命令

```bash
./test.sh
./run.sh generate --package benchmark/NP-002/asset_package.md --asset A15 --run <新运行名> --backend local --provider-config config/codex_sol.json --execute --max-iterations 12 --max-model-calls 30
scripts/run-scene-batch.sh --batch <新批次名> --protocols NP-002 --input-root benchmark --provider-config config/codex_sol.json --generation-config config/experiments/np002_openai_generation.json --scene-config config/experiments/np002_openai_scene.json --environment config/scene_environment_ffw.json --max-model-calls 150 --max-reserved-tokens 240000000 --scene-iterations 8 --asset-iterations 6
scripts/run-scene-batch.sh --batch <新批次名> --resume --execute
python3 -I scripts/analyze_scene_quality.py <运行目录或导出包的 output 目录>
```

## 9. 已实施的改进与验证（分支 `improve`，工作副本 `~/np002_agent_v2`）

| 改动 | 位置 | 说明 |
|---|---|---|
| 设计咬合不算碰撞缺陷 | `asset_generation/material.py` | 已由关节或焊接连在一起的部件之间，嵌入不超过 3 mm 记入 `engagements`，不再计为 collision；更深的嵌入和无关部件间的穿透仍是缺陷 |
| 刚性连通误报修复 | `asset_generation/material.py` | 同一刚体内的几何对原来只要在铰接层面已连通就跳过测距，导致刚性层面永远连不上，被误报为"断开构件"；现在只在刚性层面已确认连通时才跳过 |
| 可执行修复提示 | `asset_generation/material.py` | 每条碰撞给出两侧 body、穿透深度，以及在 geom 所属 body 局部坐标下的分离方向；每个断开孤岛给出最近的同刚体几何、间隙和移动方向 |
| 无法量测不阻断 | `asset_generation/pipeline.py`、`scene_generation/recovery.py`、`scene_first.py`、`review.py` | 只剩 partial（量测或探测能力不足）且没有 fail 时，单资产照常进入视觉审查；静态场景审查可 accept，结果为 `static_preview_ready`，缺口逐项列入 unverified，不算质量通过 |
| 修复比较不再一票否决 | `scene_generation/progress.py` | 检查行变得不完整只做记录，不再否决其他检查上证据完整的确认改善；检查行消失或变为 skipped 仍然否决 |
| 修复调度轮转 | `scene_generation/repair_schedule.py`、`pipeline.py` | 按资产实际被修改的次数排序，其次按尝试次数和缺陷数量；多实例任务内优先修改得少的资产 |
| 审查输出可执行 | `asset_generation/prompts.py` | finding 必须写明部件、问题、依据和建议修改；只有阻断缺陷进 findings，次要问题写入 unverified |
| 编码 agent 作者 | `asset_generation/codex_transport.py`、`selfcheck.py` | `codex_agent_author: true` 时，作者请求在资产专属工作目录里以 workspace-write 沙箱运行 codex，可反复 `./check` 自检；交付仍走原作者契约，框架独立复检；会话目录保留在运行目录的 `codex_agent_sessions/` |
| 令牌记账 | `asset_prep/storage.py` | codex 调用完成后按其报告的实际用量结算，释放发送前的保守预留；其他服务仍按预留与实报取大 |
| 不分页 | `config/experiments/np002_codex_agent_generation.json` | `review_context_bytes` 提到 3 MB，长上下文模型整包发送 |

测试：全量 1080 项，只剩原代码在 Linux 上已有的失败，本轮改动不新增失败；新增 agent 模式单测。

### 9.1 同一份 repair02 几何上的检查对比（只换检查代码，不调用模型）

| 指标 | 原代码 | 改后 |
|---|---|---|
| 111 个实例的材料完整性 | fail 51 / partial 33 / pass 27 | fail 22 / partial 34 / pass 55 |
| 碰撞对 | 2195 | 640，另有 1555 处归为设计咬合 |
| 断开构件 | 513 | 151，其中 362 个是刚性连通 bug 造成的误报 |
| 量测缺口 | 338 | 338，交给"无法量测不阻断"处理 |

### 9.2 A15 单资产（同模型 gpt-6.1-sol）

| | 原流程 | 新流程（agent 作者 + 新门槛） |
|---|---|---|
| 结局 | `needs_check_support`，卡住 | `candidate_ready`，未验证项逐条列出 |
| 模型调用 | 3 | 2（1 次作者会话，内部自检 4 次；1 次视觉审查） |
| 实报 token | 11.6 万 | 57.7 万 |

agent 会话单次 token 约为普通调用的 5 倍，但轮数大幅减少。

### 9.3 运行中发现、尚未修改

- 场景计划校验失败只回复"场景计划字段无效"，不点名多余或缺失的字段（首次计划多了 `layout_metrics`）。应在 `scene_generation/contracts.py:80` 报出具体字段。正式批次运行中，源码已冻结，下一轮再改。
