# NP-002 试跑说明

代码保持 2026 年 10 月 29 日 17:36 固定版本。只交付 NP-002 的任务资料；共享评分登记保留原文件及摘要，因此其中可能列出其他任务，但不包含其他任务的输入包。

## 文件位置

- `benchmark/NP-002/asset_package.md`：资产资料。
- `benchmark/NP-002/source.pdf`、`source.xml`：规程原文。
- `benchmark/NP-002/images/`：参考图片；实际输入以 `input_manifest.json`（输入白名单）为准。
- `benchmark/NP-002/rubric.json`：原始评分标准，未改动评分项或分值。
- `benchmark/_evaluation/generation_environment.json`：公开生成环境，属于模型可用输入。
- `benchmark/_evaluation/` 中其他文件及 `benchmark/_dev/`：评分依赖和现有评分工具，不属于模型输入。

评分标准与运行环境仍保留原文件中的“待实证验证”“部分设置尚未冻结”等状态；打包完整不等于已取得评分结果。可选的历史规则编写脚本不属于评分运行依赖，没有加入包内。

## 由接收者运行

在解压后的项目根目录打开 PowerShell（命令行工具）。本包不含虚拟环境或密钥；需先根据 requirements.txt、requirements-generation.txt 准备项目 `.venv`（虚拟环境），并在本机配置 OPENAI_API_KEY（官方服务密钥变量）。依赖与缓存建议放在 D 盘。

先准备新批次，此命令不调用模型：

```powershell
.\scripts\run-scene-batch.ps1 --batch np002_colleague_1736_01 --protocols NP-002 --input-root .\benchmark --provider-config config/openai_sol.json --generation-config config/experiments/np002_openai_generation.json --scene-config config/experiments/np002_openai_scene.json --environment config/scene_environment_ffw.json --max-model-calls 800 --max-reserved-tokens 240000000 --scene-iterations 80 --asset-iterations 60
```

确认配置和预算后，由接收者启动正式调用：

```powershell
.\scripts\run-scene-batch.ps1 --batch np002_colleague_1736_01 --resume --execute
```

使用 OpenAI（模型服务商）官方服务的 gpt-6.1-sol。不要改用历史 default.json（默认配置）。本版本批次覆盖资产、场景与质量修复，未包含后来增加的批次自动接续轨迹规划。修改代码后请另选新批次名称、重新准备，不复用旧版本冻结批次。

每规程共享预算：800 次模型调用、2.4 亿预留令牌、80 轮场景修复、每资产 60 次候选。结果在 `runs/np002_colleague_1736_01_np002`；批次汇总在 `runs/batches/np002_colleague_1736_01/index.html`。

评分时由评估方读取 rubric.json 和相关评分依赖。不要把评分文件追加到模型上下文，也不要把源码自检通过等同于评分通过。本次交付只验证输入、评分规则及机器人资源的完整性，没有运行正式实验。
