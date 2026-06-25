# DeFT 代码整理进度

> 最后更新：2026-06-25

## 进度总览

| 阶段 | 状态 | 依赖 |
|------|:--:|------|
| ① 手稿 Method 定稿 | ✅ 已完成（DCI / PRM / DCS 术语锁定） | — |
| ② 代码重命名 | ✅ 已完成（deft/ 包，3211 行，9 文件） | ① |
| ③ 模块拆分 | ✅ 已完成（DCI/PRM/DCS 文件分离） | ② |
| ④ 中文→英文翻译 | ✅ 已完成（0 中文残留） | ② |
| ⑤ 环境配置 | ✅ 已完成（requirements.txt + check_env.py） | ② |
| ⑥ README 重写 | ⬜ 待讨论 | ①⑤ |
| ⑦ 预热测试（全平台） | ⬜ 等 ⑥ | ③④⑤ |

## 命名对照

| 手稿名 | 全称 | 代码内部对应 |
|--------|------|-------------|
| **DCI** | Descriptor Conditioning Interface | `_build_degradation_descriptor()` + FiLM conditioning + degprompt bank |
| **PRM** | Polarized Route Mixture | `DualRoutePromptWrapper` + `AggressiveDenoiseRoute` + `StructurePreserveRoute` + spatial gate |
| **DCS** | Descriptor-Conditioned Scheduler | `adaptive_steps` logic in `adapt()` — σ_mad → (K, η) budget |
| DeFT | Descriptor-Forked Test-Time Adaptation | `G7_DATTA` class |

## 架构叙事

手稿锁定：**一个描述子状态的三个投影**（condition=DCI, space=PRM, update schedule=DCS），不是串行堆栈。代码应反映这一叙事——模块间是并行消费同一描述子，不是 DCI→PRM→DCS 流水线。

## 代码策略

**不修改原始脚本**（`~/ttt/code/` 内所有文件保持不动）。在 `github-code/` 目录下**从零新建**文件，参考原始代码结构重新组织。

## 路径适配约定

```bash
export DEFT_ROOT=/path/to/deft-open-source
CHECKPOINT_DIR="${DEFT_ROOT}/checkpoints"
DATA_DIR="${DEFT_ROOT}/data"
RESULTS_DIR="${DEFT_ROOT}/results"
```

## 核心文件映射

| 内部文件 | → | 目标文件 | 核心内容 |
|---------|---|---------|---------|
| `g_line_models.py` (G7_DATTA, ~6400行) | → | `deft/model.py` | DeFT 主类 + adapt() 循环 |
| `g_line_models.py` (_build_degradation_descriptor) | → | `deft/dci.py` | 6 维描述子计算 + FiLM 条件 |
| `g7_external_variants.py` (DualRoutePromptWrapper) | → | `deft/prm.py` | 极化双路 + 空间门控 |
| `g_line_models.py` (adaptive_steps 逻辑) | → | `deft/dcs.py` | 图像条件化调度 |
| `unet.py` (JointUNetMIM / DANRFUNet) | → | `deft/backbone.py` | 骨干网络 |
| `prompt_film.py` | → | `deft/adapters.py` | FiLM 包装器 |
| `g_line_models.py` (Neighbor2NeighborLoss) | → | `deft/loss.py` | 自监督损失 |

## 不需要放入的文件

- `g_line_models.py` 中 G1-G6 类 → 删除
- `patch_g7_v2.py`, `g7_ugsm_patch.py`, `g7_logic_update.py` → 删除
- `adr_loss.py`, `sspb_loss.py`, `mspc_loss.py`, `apd_loss.py` → 删除（规范配置全部关闭）
- `models/gan_wrappers.py`, `cut_patchnce.py`, `reggan.py` → 删除（G 线以外的 baseline）
- `scripts/train/` 下非 `train_G_unified.py` 的文件 → 删除
- `scripts/runners/` 下非 `run_G.sh` 的文件 → 删除

## 已完成的准备工作

| 事项 | 状态 |
|------|:--:|
| 规范 DeFT 配置精准确认 (`serial_full_eval.sh:33-42`) | ✅ |
| 手稿修订方案 (`manuscript-revision-pack.md`) | ✅ |
| 数据集路径映射（内部 Q0/Q2/Q3 ↔ Paper Q1/Q2/Q3） | ✅ |
| 补充消融实验 (No HR, K=3, Channel Gate, No MAD, Scalar Gate, sweeps) | ✅ |
| DATA_INDEX 更新 | ✅ |
| 实验路径组织 | ✅ |
| Gate Map 提取方案验证 | ✅ |
| 所有 checkpoint 上传至 Google Drive | ✅ |
| README 数据集下载部分已更新 | ✅ |
| 手稿 Method 定稿 (DCI/PRM/DCS) | ✅ |
