# TIAGo OMNI（单臂全向移动机器人）接入

默认机器人已经换成 PAL Robotics（帕尔机器人公司）公开描述中的完整 TIAGo OMNI，使用 `base_type=omni_base`（全向底盘配置）。底盘、轮组、升降躯干、头部、七关节手臂均来自厂家整机描述；装配器不再给它添加合成底盘或立柱。多桌布局和资产生成流程继续沿用现有实现。

## 资源与构型

- `data/robots/tiago_omni/model.xml`（生产装配模型）：完整移动机器人，腕部不带夹爪，由原有末端任务生成专用装置后装配。
- `data/robots/tiago_omni/stock_gripper.xml`（原厂夹爪展示模型）：同一整机，带 PAL 平行夹爪。与生产装配模型明确分开，避免双重夹爪。
- `provenance.json`（来源记录）保存六个厂家仓库的固定提交、原始资源摘要、转换说明及产物摘要；`LICENSE`（许可）保留上游许可文本。生成过程中没有调用模型。
- 原始展开描述保存在 `source_*.urdf`；URDF（统一机器人描述格式）导入参考保存在 `import_*.urdf`，用于运动学一致性测试。导入参考中的路径属于本地构建记录；正式模型仅引用自己目录内的相对资源，能够独立复制。

固定厂家版本：`tiago_robot` 为 `f1c33c92bdde7c1dd79f0c3e739e98a233dbd30b`；`omni_base_robot` 为 `49ecf709e3f42ba0613ccaeb8a07105be24a6c36`。其余依赖见来源记录及下载脚本。

## 参数和验证范围

[厂家规格](https://pal-robotics.com/wp-content/uploads/2024/04/Datasheet-TIAGo.pdf)提供 0.87 米臂展、不含末端的 3 千克全伸展负载、0.35 米升降行程；[全向底盘文档](https://docs.pal-robotics.com/edge/hardware/tiago/mobile-base-omni)提供约 720×500×310 毫米底盘尺寸。

本模型保留厂家的连杆惯性、关节轴、限位及碰撞形状。当前生产配置升降坐标为 0.20 米，肩部高度由实际运动学计算为 0.9372 米。无末端配置的 73.1860860411 千克是所选描述中的总质量，不是实测底盘质量。

模型包含 3 个布局用平面坐标、4 个轮轴、1 个躯干升降、7 个机械臂关节和2个头部关节。展示配置另有2个夹指关节。关节是实际运动学关节，不是只改变外观。

工程近似仍明确保留：

1. 平面 X/Y 移动与偏航表达底盘位姿；尚未实现麦克纳姆滚子接触、轮速运动控制、定位误差。
2. 位置伺服使用厂家力矩/力上限和工程增益，不是原厂控制器。原厂传感器外形不代表感知算法已实现。
3. MuJoCo（物理仿真引擎）对网格采用凸包碰撞；未认证完整机器人动力学或操作任务。
4. 现有可接近性建议仍采用默认肩部高度与距离，不搜索升降姿态、逆运动学或操作轨迹。
5. 展示图里的实验台布局由开发者安排，仪器来自原 NP-001 批次；不是新一轮模型实验或质量通过结果。

## 查看

在 `D:\harness\new_harness` 执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\view_tiago_omni.py
```

这是可旋转、缩放及调整关节的运动学查看器，不进行动力学步进，也不调用模型。`--flange`（腕部配置）可查看生产装配使用的无夹爪整机。

效果图与离线重装配记录在 `.cache/tiago_upgrade_20261020`。正式实验继续使用 `run.ps1`（启动脚本），使用新运行名称及最新工程验证摘要；旧运行的配置和机器人身份不会被重写。

## 重新准备资源

本机已经完成准备。`.\run.ps1 scene --prepare-robot` 会按默认环境离线校验整机文件，不会误下载 FR3（旧版弗兰卡机械臂）。固定桌面环境仍可显式选择 `config/scene_environment_fixed.json`。

在新环境恢复资源时，先准备已有项目环境，再执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --target .cache\tiago_build_deps --no-deps xacro==2.1.1 pycollada==0.9.2 PyYAML==6.0.3 python-dateutil==2.9.0.post0 six==1.17.0
.\.venv\Scripts\python.exe .\scripts\fetch_tiago_sources.py
.\.venv\Scripts\python.exe .\scripts\build_tiago_omni.py
.\run.ps1 scene --prepare-robot
```

xacro（机器人描述宏工具）、pycollada（网格格式读取库）及其辅助依赖只安装到项目缓存；构建也使用项目既有 MuJoCo、NumPy（数值计算库）和 trimesh（网格处理库）。源码按固定提交获取；已有资源只核对摘要，不自动覆盖。若上次构建留下不完整目录，先保留并检查它，再重建。

## 工程审查

第一轮核对整机/合成平台分支、平面坐标方向与顺序、腕部接口、夹爪配置、默认环境和准备入口。发现并修正了“默认环境变更后准备入口仍下载旧机器人”的问题；旧 FR3 测试改为显式读取固定环境配置。

第二轮核对资源来源、不可覆盖、历史资产与预算边界。补充固定提交、逐文件摘要、整机装配后的独立资源重载、厂家描述与转换模型在不同升降高度下的连杆位姿/关节限位/总质量一致性测试。历史场景只写新重装配目录，不改原资产、原记录、公差或账本。

全量测试首次发现 MuJoCo 文本导出会舍入物理参数：总质量出现约 0.008 克偏差。转换器现从导入模型回填完整精度的连杆位姿、惯性及关节参数；未放宽一致性断言。针对性测试已核对三个升降高度下的厂家描述与转换模型。

测试结果以 `.cache/tiago_upgrade_20261020/full_tests_final.log` 和 `.cache/verification.json`（工程验证收据）为准。测试通过不等于正式实验或动力学验收通过。
