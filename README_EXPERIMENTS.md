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
| projection_audit | audit_robot_projection.py | diagnostics/步骤名：自身部件、裁剪与 GT 轮廓核验 |
| coarse_audit | audit_coarse_integration.py | diagnostics/步骤名：冻结 VAE、opacity 和采样范围消融 |
| nccl | check_nccl.py | diagnostics/步骤名（双卡通信核验） |
| compare | export_mvwd_comparisons.py | 已有 predictions/推理步骤名 内：GT、coarse、prediction 对照 |
| review | review_mvwd_predictions.py | evaluations/步骤名/metrics.json；对照图放 reports/步骤名 |
| check | check_mvwd_checkpoint.py | evaluations/步骤名/checkpoint_finiteness.json |
| sam / views | prepare_mvwd.py | 配置中的共享缓存；终端日志及配置快照归入实验 |

`--config`、`--output`、`--mode` 由入口管理，不能在 `--` 后重设。
其他参数按原脚本传递。失败的步骤同样保留日志和退出码；重试使用新的步骤名，
例如 `a5000_robot_response_retry1`。同一实验不要并发修改 config.yaml。

新增分析脚本或评估入口时也要遵循上述布局：输入指向明确的 checkpoint/预测目录，
原始数值写 evaluations，结论及对照图写 reports；不要把分析结果写进数据集或 models。
提交 Git 的是脚本、基础配置和说明；artifacts 默认不提交。实验目录位于 workspace，
重要最终 checkpoint 和报告仍需按存储方案另做持久备份。

## 全场景长训练与多卡

基础配置：`configs/mvwd_level1_fullscene.yaml`（全部 train scenes、stride=10）。
本轮命令、资源规格和续训方式见
[全场景预实验执行说明](artifacts/mvwd/experiments/20261009_fullscene_pilot/reports/RUNBOOK.md)。

`train_mvwd.py` 新增 `--devices`、`--batch-size`、`--max-epochs`、`--max-time auto`、
`--checkpoint-every-n-steps`、`--val-every-n-batches`、`--val-episodes-per-scene`、
`--val-frame-stride`。多卡使用单节点 DDP，显式允许原流程中未使用的参数；
跨卡汇总 val 日志，模型、loss 和 optimizer 不变。`--max-time auto` 需要计算节点内
可查询的 SLURM_JOB_ID，按当时剩余时间减去保存余量；不要在登录节点运行训练。

默认每500 optimizer steps保存，保留最近两份 step 文件、last，及单独的最佳 val loss 文件。
每卡 batch×devices×accumulation 是有效 batch；改变卡数要记录，不能只比较 step 数。
`--resume` 恢复完整权重和 Adam；max-steps 是累计目标。PL 1.5 的中途恢复会重新开始
数据遍历，不是逐 batch 无缝恢复。CSV每10步flush。steps/*.json记录成功退出与中断状态。

SAM可用 `--num-shards 2 --shard-index 0/1` 并行准备；两个shards均完成后才训练。
同一configuration只分给一个writer，条件缓存配方与着色不变。

## 已有结果

既有结果建立了 [索引](artifacts/mvwd/INDEX.md)，保留原路径与原始记录，避免迁移破坏
配置、provenance、旧命令和报告链接。它们属于历史实验布局，后续不往其中追加新任务。
当前重点结果是 `night_20261008` 的训练对照及 `condition_diagnostics/review/REVIEW.md`
的条件响应诊断。以后每次完成实验，都在该实验的 `reports/REVIEW.md` 汇总状态、
checkpoint/step、样本范围、指标、限制和下一步。

## HoreKa-2 双 H200 的 NCCL 兼容设置

在 hkn1956 上，原 PyTorch 2.0.1/cu118 的 NCCL 2.14.3 首次 barrier 报
`Failed to find reverse path`，独立通信测试同样复现。集群提供的 NCCL 2.27.5
已在同一节点通过双卡 barrier、all-reduce 和 broadcast；采用进程级 preload，
不更改 Conda 环境、模型、loss 或 optimizer：

```bash
export LD_PRELOAD=/software/easybuild/x86_64/software/NCCL/2.27.5-GCCcore-13.3.0-CUDA-12.6.0/lib/libnccl.so.2
export NCCL_DEBUG=INFO
```

随后按原训练命令执行。每个新的 GPU shell 需重新设置；此路径是集群库，不适用于其他机器。
不需要把整个 CUDA/PyTorch module 加载进现有环境。该 NCCL 构建于 CUDA 12.6，
PyTorch 自身仍为 cu118；这里只采用已实测组合，不据此承诺任意版本混用。
PyTorch 2.0 的 `torch.cuda.nccl.version()` 是编译版本；实际版本见初始化日志和诊断的
`nccl_runtime`。步骤记录同时保存 NCCL_*、LD_PRELOAD 和 LD_LIBRARY_PATH。

更换节点/库后，先用唯一名称执行小型通信核验：

```bash
python scripts/run_mvwd_experiment.py run 20261009_fullscene_pilot \
  --name nccl_newnode_check --kind nccl
```

诊断测试 1、1M、16M float32 元素的求和与广播，逐元素核验两 rank 的结果。
其耗时包含初始化/同步差异，不作为通信性能 benchmark。真实训练的 20-step 检查、
续训及最新状态见本轮 RUNBOOK 和 reports/REVIEW.md。

## 自动 GT 对照

普通 `infer` 完成全部生成后，默认读取相同 episode/observer/frame 的 GT RGB，
使用与训练 adapter 相同的 BOX resize，自动保存 `GT / Coarse / Prediction` 对照。
生成阶段仍使用 `include_targets=False`，`metadata.json` 中的 `ground_truth_loaded=false`
只描述生成阶段；GT 读取记录在 `comparison.json` 中，不修改原预测或其元数据。
反事实 `diagnose` 不导出 GT 对照，因为不存在对应的反事实真值。

按用户2026-10-11的目录偏好，自动对照与预测放在同一个目录，不另建 reports 子目录：

```text
predictions/推理步骤名/
├── metadata.json / predictions.jsonl
├── comparison.json / GT_COMPARISON.md
├── bev_comparison.json / BEV_COMPARISON.md
├── episode_id/gt_env_bev.png           原始无机器人环境 BEV
├── episode_id/bev_calibration.json    env/world 原始标定
└── episode_id/robot_00/
    ├── frame_000.png                  原预测
    ├── frame_000.coarse.npz           原 coarse 数组
    ├── frame_000.world_bev.png         该物理帧真实 world BEV（含机器人）
    ├── frame_000.gt.png               同一 query 的 GT
    ├── frame_000.coarse.png           coarse RGB
    ├── frame_000.comparison.png       GT / coarse / prediction
    └── comparison_overview_01.png     按时间排列，最多六帧一页
```

`reports/` 只保留需要的实验分析和跨结果比较，不再为每次 infer 创建报告目录。
没有 GT 或希望避免额外 I/O 时传 `--no-gt-comparison`。
导出失败时预测仍保留，但步骤记录失败；可通过 CPU compare 入口补齐，
无需再进行 diffusion。已有对照文件不会被覆盖。

给已完成的预测补对照，使用新的步骤名；输出仍写进原 prediction 目录：

```bash
python scripts/run_mvwd_experiment.py run 20261009_fullscene_pilot \
  --name best_train_rs_gt_inline --kind compare -- \
  --predictions artifacts/mvwd/experiments/20261009_fullscene_pilot/predictions/best_train_rs_preview
```

删除 reports 不会解除同名 infer 的步骤保护；steps、logs、predictions 仍然存在。
真正重新推理需使用新的步骤名；仅补 GT 时使用 compare 即可。
这些是可视化产物，不代替完整 val/test 指标计算。

### 预测中的真实 BEV 对照

自动 GT 导出同时保存该 episode 的原始 before 环境 BEV（无机器人）和每个 query
对应物理帧的真实 world BEV（含机器人）。直接读取正式 NPZ 的 RGB，保留原分辨率
和颜色，不做模型输入的 letterbox、墙颜色替换，也不是额外绘制的机器人位置图。
同一物理帧的 world BEV 与 observer 无关，故不同 observer 的图片相同。
每条 episode 的 `bev_calibration.json` 记录 env/world 标定、楼层、shape 和来源；
`bev_comparison.json` 关联 query 与文件，`BEV_COMPARISON.md` 提供索引。
world BEV 仅在生成结束后读取，不作为模型条件；`--no-gt-comparison` 同时关闭这类导出。

给已有预测仅补 BEV（保留现存 perspective GT 对照）：

```bash
python scripts/run_mvwd_experiment.py run 20261009_fullscene_pilot \
  --name best_train_rs_bev --kind compare -- \
  --predictions artifacts/mvwd/experiments/20261009_fullscene_pilot/predictions/best_train_rs_preview \
  --bev-only
```
