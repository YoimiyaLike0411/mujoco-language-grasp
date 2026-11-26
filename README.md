# MuJoCo Language-Conditioned Grasping & Dexterous Hand Extension

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3.7-0EA5E9)](https://mujoco.org/)
[![Course Project](https://img.shields.io/badge/Course%20Project-AI%20Practice%202026-10B981)](#)

> **课程说明**：本项目为 **2026 年短学期《人工智能实践》课程大作业**（Course Project for *AI Practice*, Short Semester 2026），包含语言条件桌面抓取闭环系统与 16 自由度灵巧手拓展两部分实现。

---

## 项目简介（What It Does）

本项目在 **MuJoCo** 仿真环境中构建了一套面向 **Franka Emika Panda** 机械臂的语言条件抓取系统，覆盖从自然语言指令理解、视觉感知到机械臂抓取与分拣执行的完整闭环，并进一步拓展至多指灵巧手抓取场景：

1. **主模块：语言条件桌面抓取与分拣（Root）**
   - 输入自然语言任务指令、双相机 RGB-D 图像与机械臂本体观测；
   - 结合 **VLM + SAM3** 目标感知、RGB-D 空间定位跟踪与数值逆运动学（IK），实现桌面物体的抓取、搬运、精准放置与多物体分拣闭环控制；
   - 支持三阶段任务评测（基础放置、YCB 物体抓取、多物体分拣）及自动化单元测试。

2. **拓展模块：16 自由度灵巧手抓取（[`dex_grasp_ext/`](dex_grasp_ext/README.md)）**
   - 将末端二指夹爪替换为 **16 自由度四指灵巧手（MuJoCo Menagerie `wonik_allegro`）**；
   - 在完全隔离的工程结构下实现了灵巧手资产挂载、抓取笼标定、逐物体抓取参数合成、手指协同控制与闭环 Episode 验证。

---

## 仓库结构

```text
mujoco-language-grasp/
├── policies/               # 语言条件抓取闭环控制策略实现（核心入口：student_policy.py）
├── src/                    # MuJoCo 仿真环境、机器人与场景资产、评测框架（graspbench）
├── configs/                # 三阶段公开评测任务配置（Place / YCB / Sort）
├── scripts/                # 环境初始化、模型服务检查与批量验证脚本
├── tests/                  # 自动化单元测试
├── dex_grasp_ext/          # 16-DoF Allegro 灵巧手拓展工程（资产、抓取合成与评测）
├── docs/                   # 实现记录与详细说明（IMPLEMENTATION_STATUS.md）
├── outputs/                # 实验验证与评测报告归档
└── runtime/                # 无界面远端运行脚本
```

---

## 快速开始

### 主模块（Franka + 二指夹爪）

```bash
bash scripts/setup.sh
source .venv/bin/activate

# 运行单元测试
pytest

# 运行公开任务评测
python -m graspbench.evaluate --tasks configs/task1_place_public.json --output runs/task1
python -m graspbench.evaluate --tasks configs/task2_ycb_public.json --output runs/task2
python -m graspbench.evaluate --tasks configs/task3_sort_public.json --output runs/task3
```

### 灵巧手拓展模块（Franka + 16-DoF Allegro Hand）

详见 [`dex_grasp_ext/README.md`](dex_grasp_ext/README.md)：

```bash
cd dex_grasp_ext
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/01_build_assets.py --hand allegro
./.venv/bin/python scripts/02_build_scene.py --hand allegro
MUJOCO_GL=glfw ./.venv/bin/python scripts/10_eval_episodes.py
```

---

## 更多记录

- 详细实现迭代与实验边界记录见 [`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md) 及 [`outputs/`](outputs/)。
