# dex_grasp_ext — 灵巧手抓取拓展（方向 3）

课程大作业的**拓展部分**实现：把原本的 Franka + 二指夹爪，换成 **Franka + 16 自由度四指灵巧手
（MuJoCo Menagerie `wonik_allegro`）**，重做抓取合成、手指协同控制与抓取验证，用于研究
"多指手能不能在同一个语言—视觉—抓取闭环里完成抓取/分拣"。

本目录是一个**完全隔离**的工程：不修改课程项目（`mujoco-grasp-assignment` 与
`rlx/mujoco-grasp-rebuild`）里的任何文件，也不改动它们的 Python 环境。

## 隔离做法

| 项目 | 课程工程 | 本拓展 |
|---|---|---|
| 机器人 | Franka + 二指夹爪 | Franka + Allegro 灵巧手（同一夹爪法兰） |
| 场景 | `graspbench/assets/scene.xml` | 由脚本**只读读取**课程场景后生成 `assets/scene_dex.xml` |
| 依赖 | 课程 `.venv` | 本目录自己的 `.venv`（`mujoco==3.3.7`，与课程同版本） |
| 代码 | `policies/student_policy.py` | 本目录 `dexgrasp/`，不 import 课程策略代码 |

生成脚本只做两件事：把课程场景里的机器人 `include` 换成我们的机器人，并把相对资产路径改成
绝对路径；课程文件始终保持只读。

## 目录

```
assets/vendor/franka_emika_panda/   课程 Panda 资产的副本（含许可证）
assets/menagerie/                   MuJoCo Menagerie 的 wonik_allegro / leap_hand
assets/robot/                       生成的片段与 panda_allegro.xml（挂载灵巧手）
assets/scene_dex.xml                生成的完整场景（桌子/托盘/物体/相机与课程一致）
dexgrasp/                           扩展代码：IK、手部协同、抓取实验
scripts/                            资产生成、探针、抓取实验与演示
runs/                               实验结果与演示视频
```

## 复现

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/01_build_assets.py --hand allegro   # 生成手部片段 + 挂载
./.venv/bin/python scripts/02_build_scene.py --hand allegro    # 从课程场景生成 scene_dex.xml
MUJOCO_GL=glfw ./.venv/bin/python scripts/03_scene_probe.py     # 检查挂载与可达性
MUJOCO_GL=glfw ./.venv/bin/python scripts/09_calibrate_cage.py  # 标定抓取笼
MUJOCO_GL=glfw ./.venv/bin/python scripts/08_grasp_synthesis.py # 逐物体搜索抓取参数
MUJOCO_GL=glfw ./.venv/bin/python scripts/10_eval_episodes.py   # 跑闭环 episode 评测
```

感知需要 GPU 机上的 SAM3 服务，本机通过 SSH 端口转发使用（主机名/账号请替换成自己的）：

```bash
ssh -N -L 8766:127.0.0.1:8766 <user>@<gpu-host>   # 校内网需先建立 VPN（如 zju-connect）
```

## 已实现

1. **挂载求解**（`scripts/01_build_assets.py`）：把灵巧手挂到课程 Panda 的法兰上。手掌法向对齐
   夹爪工具轴（掌心朝下）、手指水平，形成桌面电源抓取构型；松开 0.042 m 让手掌让开法兰，
   实测与机械臂无自碰撞。
2. **场景生成**（`scripts/02_build_scene.py`）：与课程完全相同的桌面、托盘、物体与相机，
   只替换末端执行器。
3. **本拓展自带的 DLS IK**（`dexgrasp/ik.py`）：以手掌坐标系为目标，位置误差 <2 mm。
4. **抓取笼位姿**（`dexgrasp/hand.py`）：以"拇指指尖与三指指尖中点"定义抓取笼中心，
   并测量了不同闭合力下的指尖间距。
5. **抓取实验**（`dexgrasp/experiment.py`）：接近 → 下降 → 合拢 → 抬升，统计抬升高度、
   物体漂移、手指接触数。
6. **真实感知**（`dexgrasp/perception.py`）：MuJoCo 渲染 RGB-D → SAM3 HTTP 服务 →
   掩码解码 → 反投影到世界坐标；与真值对比误差 0.7–3.5 mm（紧凑物体）。
7. **抓取合成**（`dexgrasp/planning.py` + `scripts/08/09`）：先用"拇指对三指"的对跖几何
   标定出"闭合力 → 可容纳半径 / 笼心"，再对每个物体在仿真中搜索闭合度与腕部偏航，
   以抬升高度选参数。
8. **完整闭环**（`dexgrasp/pick_place.py`）：感知目标与托盘 → 高位中转 → 接近 → 合拢 →
   抬升 → 搬运 → **边开爪边上升**释放 → 退回 → 复检物体是否落在托盘内。

## 目前结果（macOS，CPU，MuJoCo 3.3.7，真实 SAM3）

**闭环 episode（`scripts/10_eval_episodes.py`，6 个物体各 1 次）**

| 物体 | 目标托盘 | 结果 | 抬升 | 放置偏差 | 感知误差 |
|---|---|---|---|---|---|
| apple | 方盘 | 成功 | 206 mm | 41 mm | 3.3 mm |
| orange | 圆盘 | 成功 | 222 mm | 72 mm | 3.5 mm |
| red_cube | 方盘 | 成功 | 203 mm | 55 mm | 0.8 mm |
| green_cylinder | 圆盘 | 成功 | 218 mm | 68 mm | 0.7 mm |
| potted_meat_can | 方盘 | 成功 | 217 mm | 54 mm | 1.2 mm |
| banana | 圆盘 | 失败 | — | — | 27.7 mm |

合计 **5/6**（每个物体 1 次；感知与放置全部使用真实 SAM3 + 自研几何，未使用真值位置）。

**重复评测（6 物体 × 3 出生点）**：开启"抬升失败后换抓取参数重试"后为 **12/18（67%）**。
失败分类见 `docs/STATUS.md`：其中 7 次是 SAM3 没分割出目标/托盘/复检目标（感知召回问题），
只有 1 次是抓取本身没抬起。

**抓取合成表（`runs/grasp_synthesis/grasp_table.json`）**：每物体 18 个候选（6 档闭合度 × 3 个偏航）。

| 物体 | 最优闭合度 / 偏航 | 成功候选数 |
|---|---|---|
| apple | 0.55 / 20° | 9/18 |
| orange | 0.55 / 20° | 10/18 |
| red_cube | 0.65 / 20° | 8/18 |
| green_cylinder | 0.55 / 0° | 14/18 |
| potted_meat_can | 0.70 / 20° | 18/18 |
| mustard_bottle | 0.65 / 0° | 16/18 |
| banana | 0.55 / 0° | 0/18 |

两个关键发现：

1. 灵巧手的成功率对闭合力是**非单调**的（接触构型会突变），所以必须按物体做抓取合成，
   固定闭合度不可用；
2. 腕部偏航 20° 对多数物体更好——手的拇指与三指是不对称的，"手掌正对物体"并不是最优姿态。

演示视频：`runs/pick_place_orange/episode.mp4`（橙子入圆盘）、`runs/pick_place_apple/episode.mp4`。

## 后续（见 docs/STATUS.md）

- 接真实 SAM3 感知（HTTP 服务在实验室内网 GPU 机上）替换真值物体位姿；
- 抓取合成：按物体尺寸在仿真中搜索闭合度/偏置/偏航，给出抓取质量评分；
- 搬运到托盘 + 放置后验证，构成完整"感知—规划—执行—验证"闭环；
- 与二指夹爪基线做同任务对照，输出统计与失败分类。

## 许可

`assets/vendor/franka_emika_panda`（Apache-2.0，Google DeepMind MuJoCo Menagerie）与
`assets/menagerie/*`（各模型自带 LICENSE）均保留原始许可证文件。
