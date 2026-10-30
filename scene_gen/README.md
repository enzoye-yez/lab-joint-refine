# NP-002 工程交接与运行说明

解压后可以直接查看和修改代码。运行实验前，需要安装 Python（编程语言）3.11、建立虚拟环境、安装依赖，并配置接收者自己的官方服务密钥。

## 环境准备

建议解压到 D 盘。在 PowerShell（命令行工具）中执行，第一行替换为实际解压路径。不需要激活虚拟环境。

```powershell
Set-Location D:\experiments\np002_agent_20261029_1736_with_np002
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir .\.cache\pip -r requirements.txt -r requirements-generation.txt "numpy==2.4.6"
```

依赖和下载缓存放在项目目录内。依赖文件固定了 MuJoCo（物理仿真引擎）3.3.7。若 `py` 不可用，将 `py -3.11` 替换为已安装的 Python 3.11 解释器完整路径。

接收者需自行配置 `OPENAI_API_KEY`（官方服务密钥环境变量），确保运行命令的进程能够读取。不要把密钥写入源码或重新分发的压缩包。以下命令明确选择 OpenAI（模型服务商）官方配置，模型为 `gpt-6.1-sol`，地址为 `https://api.openai.com/v1`。

## 准备并启动 NP-002

先准备新批次，**这一步不调用模型**：

```powershell
.\scripts\run-scene-batch.ps1 --batch np002_colleague_1736_01 --protocols NP-002 --input-root .\benchmark --provider-config config/openai_sol.json --generation-config config/experiments/np002_openai_generation.json --scene-config config/experiments/np002_openai_scene.json --environment config/scene_environment_ffw.json --max-model-calls 800 --max-reserved-tokens 240000000 --scene-iterations 80 --asset-iterations 60
```

确认配置与预算后，由接收者执行以下命令。**这一步会调用模型并产生服务费用**：

```powershell
.\scripts\run-scene-batch.ps1 --batch np002_colleague_1736_01 --resume --execute
```

每规程共享预算为 800 次模型调用、2.4 亿预留令牌、80 轮场景修复、每资产 60 次候选。这些是上限，不是预计消耗；恢复和回退不重置已发生消耗。

- 实验结果：`runs/np002_colleague_1736_01_np002/`。
- 批次汇总：`runs/batches/np002_colleague_1736_01/index.html`。

上述命令覆盖资产生成、场景装配和质量修复。源码虽有机器人操作模块，**这个历史版本没有后来增加的批次自动接续轨迹规划功能**，不支持后来新增的 `--with-operation` 参数。场景能加载、质量通过和机器人任务执行成功需要分别判断。

修改代码或配置后，请换一个新批次名称重新准备，并在启动命令中使用相同的新名称，不绕过旧批次的版本检查。

## 输入与评分边界

模型输入以 `benchmark/NP-002/input_manifest.json` 的白名单为准。公开生成环境位于 `benchmark/_evaluation/generation_environment.json`。`rubric.json` 和其他评分依赖供评估方使用，不要追加到模型上下文。

评分资料保留原来的“待实证验证”和“部分设置尚未冻结”状态。本次打包没有执行评分或正式模型实验，也没有在接收者机器上验证依赖安装、网络和账号权限。

## 历史版本与来源

这是用户确认的“发送旧场景查看命令时”已固定的工程版本，不是生成 repair02 场景的更早版本。

源码摘要：`5a3f445268afee07170d45339c4b6bdcc8f4699dc0b77d0346e771916eecf4e4`。

src、tests、scripts、docs、run.ps1 和所附 config 来自 `.cache/quality_launch_20261029/project` 的实体快照；已核对当时收据。原厂机器人资源来自同一快照。安装依赖声明、测试入口、容器构建文件和工程约定作为配套文件从当前项目复制，逐项来源写入 provenance.json（来源说明），不宣称它们属于原冻结清单。

目录可用于继续工程开发。包中不含密钥、虚拟环境、实验账本、模型响应、生成资产或编译缓存。因此，旧的查看命令不能仅凭此源码包显示实验场景；需要另行交付对应场景和资源。项目启动入口为 run.ps1，已有文档位于 docs。

所附 config/openai_sol.json 明确指向 OpenAI（模型服务商）官方 gpt-6.1-sol。config/default.json 保留历史默认值，并不是官方配置；正式运行需显式选用官方配置，密钥由接收者自行配置，不在包内。

本次只核对恢复后的文件与历史快照一致，没有启动实验，也没有认证旧版缺陷已经解决。该版保留当时的报告大小、质量审查调度等已知限制；后来新增的批次自动接续轨迹规划不在此版本中。任何修改后的实验须新建运行名称，不能改写或绕过旧实验的版本绑定。

## NP-002 资料已加入

本交付包新增 benchmark/NP-002 下的原文、资产清单、参考图片、输入白名单和 rubric（评分标准），并包含其公共环境与评分依赖。操作说明及完整试跑命令见 NP002_TRY.md。源码版本未改变；本包仍不包含密钥、历史实验记录或生成场景。

共享评分登记可能提及其他任务，本包只包含 NP-002 的任务输入。文件摘要见 `file_manifest.json`（文件清单），输入核验见 `np002_bundle_receipt.json`（资料核验收据）。

交接后由师兄准备本地环境、阅读和修改代码，并按上面的命令运行新实验。
