# MVWD 实验产物约定

用户要求之后的结果与日志有条理地集中保存。处理本仓库的训练、推理、诊断或评估时：

- 先读 README_EXPERIMENTS.md，沿用 scripts/run_mvwd_experiment.py 的目录约定。
- 新实验放 artifacts/mvwd/experiments/YYYYMMDD_目的/；名称说明实际目的。
- 同一实验的配置、终端日志、步骤记录、checkpoint、预测、评估与报告集中在该目录。
- 使用唯一步骤名。失败和重试也保留日志；不要覆盖、混写或悄悄删除旧结果。
- 每次执行保存 argv、配置快照、环境/SLURM 信息、退出码。现有统一入口自动完成这些操作。
- 新诊断/评估脚本应接入统一入口（需要时扩展 kind 映射），不要另建随意命名的输出根目录。
- 原始指标放 evaluations，结论与对照图放 reports；完成后在 reports/REVIEW.md 记录
  checkpoint/step、数据范围、验证结果、限制及下一步。
- 历史目录按 artifacts/mvwd/INDEX.md 追溯，后续不要在其中追加新实验。
  未经明确要求不要搬迁历史数据，以免破坏配置与 provenance 路径。
- SAM/数据读取缓存共用；models 只保存权重，dataset 目录不保存实验输出。
- 在用户已经提供的交互式 Slurm allocation 内执行调试，不自行提交 debug sbatch。

此约定只规范实验管理，不授权改变模型、loss、optimizer 或已有 checkpoint。
