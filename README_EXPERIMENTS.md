# MVWD 实验结果与日志管理

后续实验统一放在本仓库 `artifacts/mvwd/experiments/`，使用
`scripts/run_mvwd_experiment.py` 创建并执行。不要再新增散落的 `night_*`、
`debug_logs` 或独立 `predictions/*` 目录。同一个实验的训练、预测、诊断、评估和报告
放在同一实验目录里；不同任务采用不同步骤名，可在两份交互 GPU allocation 上并行。

## 目录规则

实验名采用 `YYYYMMDD_目的`，例如 `20261009_control_condition_audit`。
日期按实际工作日期填写；名称描述目的，不只写 taskA/taskB。每一步的名称也要描述操作，
例如 `a5000_g9_robot_response`、`train_rs_20steps`、`val_step10000_g1`。

```text
artifacts/mvwd/
├── INDEX.md                         既有结果与新实验入口
├── experiments/
│   └── YYYYMMDD_目的/
│       ├── manifest.json            实验目的、创建时间、Git 状态、Python 路径
│       ├── source_config.yaml       创建时的原配置副本
│       ├── config.yaml              后续步骤使用的配置（绝对路径）
│       ├── steps/
│       │   ├── 步骤名.json           实际 argv、状态、退出码、SLURM/GPU 环境
│       │   └── 步骤名.config.yaml    该步骤启动时的配置快照
│       ├── logs/步骤名.log           stdout + stderr，不再手动另建日志目录
│       ├── training/步骤名/          checkpoints/、Lightning logs/、run_metadata.json
│       ├── predictions/步骤名/       图片、coarse、metadata.json、predictions.jsonl
│       ├── diagnostics/步骤名/       标定、机器人投影、条件响应等诊断
│       ├── evaluations/步骤名/       权重有限性检查、指标与评估产物
│       └── reports/                 REVIEW.md、对照图、最终结论
├── sam/                             共享 SAM 条件缓存
└── cache/                           共享数据读取缓存
```

`logs/*.log` 是完整终端日志，训练目录中的 Lightning `logs/version_*/metrics.csv`
是 loss 表，两者用途不同。步骤返回成功只代表命令退出码为 0，不代表训练收敛。
`steps/*.json` 的 running 状态若长期保留，应结合 Slurm 状态和日志确认；节点被强制终止时，
Python 来不及写 finished 状态。

模型初始化权重及 SAM 权重仍在 `models/`；dataset 和机器人资产仍在 workspace 的
`datasets/`。共享缓存的位置记录在实验配置里；旧任务的专用缓存不移动。

## 在交互节点执行

继续使用已有的 tmux + salloc/srun 流程；这个入口不会提交 Slurm 作业或申请 GPU。
进入交互节点，激活环境并进入 repo：

```bash
source ~/.config/multiview/storage.sh
conda activate top2pano
cd ~/multiview_baselines/top2pano
```

先创建实验。这个操作只建立目录，不训练或推理，也不占用 GPU：

```bash
python scripts/run_mvwd_experiment.py create 20261009_control_condition_audit \
  --config configs/mvwd_level1_robots_overfit.yaml \
  --purpose '核验已有 checkpoint 的机器人条件分支；保留原训练算法'
```

运行一步诊断。`--` 后面直接传原脚本参数；入口会自动补齐配置和输出路径：

```bash
python scripts/run_mvwd_experiment.py run 20261009_control_condition_audit \
  --name a5000_robot_response --kind diagnose -- \
  --checkpoint artifacts/mvwd/night_20261008/taskA/run/checkpoints/last.ckpt \
  --split train --scene Rs_int --max-episodes 2 --robots 1 2 --frames 10 \
  --guidances 1 9
```

这里是命令用法示例；已有条件响应结果已经完成，不需要重新生成。
使用旧 checkpoint 时，请在 create 的 `--config` 中指定与该 checkpoint 匹配的配置，
例如 `artifacts/mvwd/night_20261008/taskA/config.yaml` 或 taskB 的配置。

训练和预测也通过同一入口。例如创建一个独立的链路检查实验：

```bash
python scripts/run_mvwd_experiment.py create YYYYMMDD_robots_smoke \
  --config configs/mvwd_level1.yaml --purpose '20-step 带机器人链路检查'

python scripts/run_mvwd_experiment.py run YYYYMMDD_robots_smoke \
  --name train_rs_20steps --kind train -- \
  --scene Rs_int --max-episodes 2 --frame-stride 10 --max-steps 20 --no-validation

python scripts/run_mvwd_experiment.py run YYYYMMDD_robots_smoke \
  --name train_step20_g9 --kind infer -- \
  --checkpoint artifacts/mvwd/experiments/YYYYMMDD_robots_smoke/training/train_rs_20steps/checkpoints/last.ckpt \
  --split train --scene Rs_int --max-episodes 1 --robots 0 1 2 --frames 0 10 20
```

将 `YYYYMMDD` 换成工作日期。创建后可编辑实验 `config.yaml` 设置正式预算、guidance 等；
每一步都会保存自己的配置快照，已完成步骤不会被改写。
`--resume` 仍传给原训练脚本；续训使用新的步骤名，旧训练目录保留。

支持的 `--kind`：

| kind | 原入口 | 输出位置 |
| --- | --- | --- |
| train | train_mvwd.py | training/步骤名 |
| infer | infer_mvwd.py | predictions/步骤名 |
| diagnose | diagnose_robot_conditions.py | diagnostics/步骤名 |
| inspect | inspect_mvwd.py | diagnostics/步骤名 |
| robots | inspect_robot_rendering.py | diagnostics/步骤名 |
| check | check_mvwd_checkpoint.py | evaluations/步骤名/checkpoint_finiteness.json |
| sam / views | prepare_mvwd.py | 配置中的共享缓存；终端日志及配置快照归入实验 |

`--config`、`--output`、`--mode` 由入口管理，不能在 `--` 后重设。
其他参数按原脚本传递。失败的步骤同样保留日志和退出码；重试使用新的步骤名，
例如 `a5000_robot_response_retry1`。同一实验不要并发修改 config.yaml。

新增分析脚本或评估入口时也要遵循上述布局：输入指向明确的 checkpoint/预测目录，
原始数值写 evaluations，结论及对照图写 reports；不要把分析结果写进数据集或 models。
提交 Git 的是脚本、基础配置和说明；artifacts 默认不提交。实验目录位于 workspace，
重要最终 checkpoint 和报告仍需按存储方案另做持久备份。

## 已有结果

既有结果建立了 [索引](artifacts/mvwd/INDEX.md)，保留原路径与原始记录，避免迁移破坏
配置、provenance、旧命令和报告链接。它们属于历史实验布局，后续不往其中追加新任务。
当前重点结果是 `night_20261008` 的训练对照及 `condition_diagnostics/review/REVIEW.md`
的条件响应诊断。以后每次完成实验，都在该实验的 `reports/REVIEW.md` 汇总状态、
checkpoint/step、样本范围、指标、限制和下一步。
