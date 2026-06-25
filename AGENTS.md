# DeFT Open-Source Code — 整理项目指引

> **目标**：将 DeFT 代码从内部实验仓库整理为可公开发布的开源项目。
> **原则**：所有命名以手稿 `manuscript.md` 和 `WRITING_GUIDE.md` 中锁定的术语为准。
> **状态**：deft/ 包已完成（DCI / PRM / DCS），环境配置就绪。等待 README 重写。

## 命名对照

| 手稿名 | 全称 | 代码内部名 | 角色 |
|--------|------|-----------|------|
| **DCI** | Descriptor Conditioning Interface | `_build_degradation_descriptor` + FiLM | 冻结退化描述子 + 条件接口 |
| **PRM** | Polarized Route Mixture | `DualRoutePromptWrapper` + `AggressiveDenoiseRoute` + `StructurePreserveRoute` | 极化双路空间混合 |
| **DCS** | Descriptor-Conditioned Scheduler | `adaptive_steps` 三档逻辑 | 图像条件化更新调度 |

## 依赖顺序

```
① 手稿 Method 定稿 → ② 代码重命名 → ③ 模块拆分 → ④ 英文翻译 → ⑤ 环境配置 → ⑥ README → ⑦ 预热测试
已完成：①②③④⑤ | 剩余：⑥⑦
```

## 代码策略

**不修改原始脚本**（`~/ttt/code/` 内所有文件保持不动）。在 `github-code/` 目录下**从零新建**文件，参考原始代码结构重新组织。内部实验仓库和开源发布仓库完全分离。

## 路径适配约定

用户不可能有 `/home/zhanpeng/ttt/`。开源代码使用环境变量：

```bash
# 用户设置（可选，脚本自动检测）
export DEFT_ROOT=/path/to/deft-open-source

# 脚本自动检测（脚本顶部）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFT_ROOT="${DEFT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# 所有路径从 DEFT_ROOT 派生
CHECKPOINT_DIR="${DEFT_ROOT}/checkpoints"
DATA_DIR="${DEFT_ROOT}/data"
RESULTS_DIR="${DEFT_ROOT}/results"
```

Python CLI 已有 `--q-dataset-dir`、`--denoiser-checkpoint` 等参数，路径适配只需约定脚本中的默认值。

## 目录结构（目标）

```
deft-open-source/
├── README.md
├── LICENSE
├── environment.yaml / requirements.txt / setup.py
├── deft/                  # DeFT 核心包（从 code/models/ 提取重写）
│   ├── model.py           # DeFT 主类 (← G7_DATTA)
│   ├── dci.py             # 退化描述子 + 条件接口 (← _build_degradation_descriptor + FiLM)
│   ├── prm.py             # 极化双路混合 (← DualRoutePromptWrapper)
│   ├── dcs.py             # 描述子条件化调度 (← adaptive_steps 三档逻辑)
│   ├── backbone.py        # DANRFUNet / U-Net
│   ├── adapters.py        # FiLM / LoRA / PromptFiLM
│   └── loss.py            # N2N + Charbonnier
├── core/                  # 基础工具（从 code/core/ 复制+翻译）
├── data/                  # 数据加载（从 code/data/ 复制+翻译）
├── evaluation/            # 评估框架（从 code/evaluation/ 复制+翻译）
├── utils/                 # 噪声估计等（从 code/utils/ 复制+翻译）
├── scripts/               # 一键运行脚本（从 code/scripts/ 提取+路径适配）
├── tools/                 # 构建/预处理工具（从 code/tools/ 复制必需品）
└── checkpoints/           # 用户下载 checkpoint 后放这里
```

## 参考文件（最终权威）

- 手稿：`../manuscript.md`（§II Method, §III Experiments）
- 写作指南：`../WRITING_GUIDE.md`（Canonical Facts: DCI/PRM/DCS）
- 图表规划：`../FIGURE_TABLE_PLAN.md`
- 数据溯源：`../DATA_INDEX.md`

## 禁止事项

- ❌ 不修改 `~/ttt/code/` 下任何文件（内外分离）
- ❌ 不放 G1-G6 遗留代码、消融变体、补丁脚本
- ❌ 不放任何包含中文注释的文件（需翻译后放入）
- ❌ 不放内部路径（/home/zhanpeng/ttt/）
- ❌ 不使用旧模块名（DFP/PAM/EPG）——必须用 DCI/PRM/DCS

## 当前进度

详见 `./PROGRESS.md`。
