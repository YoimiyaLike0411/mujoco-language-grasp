> 随机抽样补测：12/15 通过，其中 Task 3 2/5。随机布局下仍存在失败；详见 outputs/RANDOM_SEED_REPORT_20260915_1.md。此前 30/30 是另一组已测案例，不能据此声称任意随机种子稳健。

> 远端无界面运行可使用 `runtime/run.sh`；版本与实测证据见 `IMPLEMENTATION_STATUS.md`。

# MuJoCo 语言条件桌面抓取 Assignment

当前为重建控制器 V3：真实 SAM3 初始化、RGB-D 水果跟踪、独立横向下降修正、放置后表面位置验证，以及自适应安全抬升。
54 项自动测试通过。种子 977989 的 Task 3 完整成功（2809/3000 步、零不安全接触）；边缘种子 874039 会先尝试 0.64 m，失败后选择 0.50–0.64 m 之间最高可达高度，罐头实测约 0.54 m，内收后再升到 0.64 m。该种子的整局结果仍为 3/4，剩余问题是边缘橙子长距离夹持滑落。已测种子与性能边界请见
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

你将实现一个从自然语言任务到抓取执行与结果验证的闭环系统。框架提供 Franka Panda、
桌面、简单形体与带材质的真实物体、双相机 RGB-D 与本体观测、数值 IK 工具、结构化 policy contract、日志、
录像和公开评测；需要完成的入口是 `policies/student_policy.py`。

## Quick start

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m graspbench.cli --output runs/inspect
pytest
python -m graspbench.evaluate --tasks configs/task1_place_public.json --episodes 1 --output runs/smoke
```

原始 starter policy 仅保持安全姿态；本目录的 `policies/student_policy.py` 已替换为闭环控制器。

模型路线必须接入真实 VLM/SAM/VLA 服务。运行随包提供的 OpenAI-compatible VLM+SAM3 模板前先执行：

```bash
python scripts/check_model_services.py
```

学生笔记本无需承载模型权重；可以使用课程服务器或商用在线服务。固定坐标和简单图像通道
判断不算模型识别。

三阶段公开评测：

```bash
python -m graspbench.evaluate --tasks configs/task1_place_public.json --output runs/task1
python -m graspbench.evaluate --tasks configs/task2_ycb_public.json --output runs/task2
python -m graspbench.evaluate --tasks configs/task3_sort_public.json --output runs/task3
```

Task 3 只有所有物体都进入正确盘子并稳定保持才算完整成功。

机器人模型来自 Google DeepMind
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)，
许可证和本地改动记录在 `src/graspbench/assets/vendor/`。课程固定 MuJoCo 3.3.7 以保证复现。

不要提交 API key、`.venv`、大模型权重或整个 `runs/`。提交代表性日志、汇总结果与演示视频即可。
