# MVWD Level 1 评估

当前 Top2Pano-Persp 是 Level 1：给定 BEV、机器人/相机状态，独立生成多个观察者图像。
抽取多个物理时刻时，每个时刻仍是独立的多视角条件，不因此成为 Level 3 视频模型。
本实现仅实现本任务所需的四个 Level 1 主指标；不混入 L2 edit/keep 或 L3 temporal 指标。

## 固定协议

`configs/mvwd_metrics_level1.yaml` 是评估协议，与模型训练配置分开。

| 项目 | 固定值 |
| --- | --- |
| 数据 | 正式 test split 全部 3338 episodes、7 scenes |
| 帧 | 每 episode 固定分层采样2帧，前/后半段各1帧；ID hash/seed42，与两帧训练的选择函数相同 |
| 观察者 | robot0/1/2，每 episode 6张图，每 seed 20028张 |
| 随机采样 | 3个生成 base seeds：42、43、44；不做 best-of-K |
| 目标 RGB | 完整视野256×448，GT用PIL BOX缩放，与训练adapter一致 |
| PSNR ↑ | RGB [0,1]，逐图先算PSNR，再做seed/episode聚合；不对RGB作min/max归一化 |
| LPIPS ↓ | 官方 `lpips==0.1.4`，AlexNet，LPIPS v0.1，冻结预训练权重，RGB转[-1,1] |
| FID ↓ | 官方 `clean-fid==0.1.35`，clean resize、Inception V3 2048维；每seed整个图像集合算一次，再对3个FID取均值 |
| MEt3R ↓ | 官方固定commit，MASt3R+DINO16+FeatUp，cosine；完整FOV双线性antialias缩放到256×256，再转[-1,1] |
| MEt3R pair | 同一episode、同一物理帧、不同机器人；固定按robot ID递增，使用官方有向score |
| GT pair资格 | 双向可见表面占比的最小值 >0.1；相机/GT depth重投影与目标深度一致性检查 |
| 可见性容差 | max(0.05m, 0.01×depth)，图像边界排除2px，并去掉深度不连续/invalid相邻像素 |
| 配对指标聚合 | episode/seed内部平均所有有效view或pair；seed均值；最后episode等权均值 |
| 95% CI | 固定seed20261015、2000次episode bootstrap；不把frame或seed当独立episode |

GT几何只选择MEt3R候选对；MEt3R自身仍用生成图像推断几何，未用GT替换官方算法。
重叠包含同一时刻的所有有效GT表面；不以生成图像/预测depth筛pair。
没有合格pair的episode记为undefined并明确报告数量，不填0。官方MEt3R若对GT合格pair
产生空重叠/非有限值时，逐pair记录undefined并将该次总MEt3R标为null，同时报告失败比例；
其他指标继续完成。不填0、不仅平均重建成功的pairs、不按生成结果重选GT pair集。
这是本实现额外明确的失效协议，尚不能把有失败的MEt3R写成主表数值。

FID不计算episode均值或episode bootstrap CI；报告每seed的全局FID及三seed均值。
少量preview图像的FID协方差可能秩亏，所有小样本结果明确标为diagnostic。
PSNR完美匹配值为+inf，JSON保存为字符串`Infinity`，不会任意封顶；含无穷值的CI不计算。
只有一个episode的paired指标没有可估计的episode-bootstrap CI。

官方实现来源：

- LPIPS: https://github.com/richzhang/PerceptualSimilarity
- Clean-FID: https://github.com/GaParmar/clean-fid
- MEt3R: https://github.com/mohammadasim98/met3r ，commit `ee0e1752898559e1a3e85e2e151d3edeb9b55f73`
- MEt3R依赖的FeatUp: commit `6b5a6c0e91f75e69194807128dcbc39c3084a30d`
- PyTorch3D: v0.7.7，commit `89653419d0973396f3eff1a381ba09a07fffc2ed`

MASt3R不是论文早期DUSt3R变体，采用官方当前默认的MASt3R配置；其他方法必须使用同一个
协议/权重/预处理。结果记录官方源码校验、包版本和后端设置，不与其他MEt3R变体混报。

## 环境

指标使用独立的 `mvwd-metrics`，不更改原 `top2pano` 环境。当前集群的环境位于 workspace
`envs/mvwd-metrics`，源码放 `software/repos/{met3r,featup,pytorch3d}`；不放进baseline repo。
生成仍用 `top2pano`，评估切换环境后运行同一个repo中的脚本。

```bash
source ~/.config/multiview/storage.sh
conda activate mvwd-metrics
cd ~/multiview_baselines/top2pano
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
```

新机器复现环境：

```bash
conda env create -f environment_metrics.yaml
conda activate mvwd-metrics
python -m pip install -r requirements_metrics.txt
```

然后把三个官方库clone到你选定的software目录，固定commit并初始化递归子模块：

```bash
git clone https://github.com/mohammadasim98/met3r.git /YOUR/SOFTWARE/met3r
git -C /YOUR/SOFTWARE/met3r checkout --detach ee0e1752898559e1a3e85e2e151d3edeb9b55f73
git -C /YOUR/SOFTWARE/met3r submodule update --init --recursive

git clone https://github.com/mhamilton723/FeatUp.git /YOUR/SOFTWARE/featup
git -C /YOUR/SOFTWARE/featup checkout --detach 6b5a6c0e91f75e69194807128dcbc39c3084a30d

git clone https://github.com/facebookresearch/pytorch3d.git /YOUR/SOFTWARE/pytorch3d
git -C /YOUR/SOFTWARE/pytorch3d checkout --detach 89653419d0973396f3eff1a381ba09a07fffc2ed
```

编译需要匹配PyTorch的CUDA toolkit和C++编译器。当前H100环境使用CUDA12.6 toolkit、
PyTorch2.4.1/cu124、GCC11；以下arch9.0只适用于H100/H200，新机器应按实际GPU设置。

```bash
export CUDA_HOME=/software/easybuild/x86_64/software/CUDA/12.6.0
export PATH="$CUDA_HOME/bin:$PATH"
# HoreKa toolkit headers/libraries live under targets, rather than the root include/lib64.
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CUB_HOME="$CUDA_HOME/targets/x86_64-linux/include"
export TORCH_CUDA_ARCH_LIST=9.0
export FORCE_CUDA=1
export MAX_JOBS=8
python -m pip install --no-build-isolation --no-deps /YOUR/SOFTWARE/pytorch3d /YOUR/SOFTWARE/featup
python -m pip install --no-build-isolation --no-deps -e /YOUR/SOFTWARE/met3r
```

以上为独立metric环境的依赖编译；不需要把编译环境的LD_LIBRARY_PATH注入原训练环境。
权重下载保存在本repo的 `models/metrics/`（gitignored）；MEt3R几何权重保存在已有HF_HOME，
两者都在workspace。结果写evaluations，不会为指标复制一份image数据集。

## 先评估已有preview

已有Pomaria2是1个episode、3个时刻×3个view、单seed42，不符合完整benchmark。
显式加`--preview`才接受这个范围；输出仍包含四指标，但不能作为完整test主表结果。

```bash
python scripts/run_mvwd_experiment.py run 20261013_fullscene_2frames_gpu4 \
  --name metrics_pomaria2_preview_new --kind metrics_level1 -- \
  --predictions artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/predictions/best_test_pomaria2 \
  --preview --device cuda
```

只核验轻量指标时可显式`--preview --metrics psnr`，不加载torch/其他模型。
若缺依赖、权重或发生运行异常，命令返回失败并保留日志；不输出伪造或替代指标。
MEt3R重建无效属于可记录的metric undefined，评估流程可完成，但`all_requested_metrics_defined`
和`benchmark_ready`为false；completed不等于四个分数都可用。
可加`--met3r-gt-reference`额外核验同一GT相机对的官方重建，参考值仅用于诊断，不混入生成指标。

## 完整test、K=3

先在训练环境生成三个固定seed的完整test预测。不传frames，沿用当前config每条2帧。
以下是需要真实GPU预算的完整推理命令，不会被指标脚本自动启动。
关闭GT预览导出以减少重复图片，评估会从dataset直接读取严格配对的GT。

```bash
conda activate top2pano
for seed in 42 43 44; do
  python scripts/run_mvwd_experiment.py run 20261013_fullscene_2frames_gpu4 \
    --name "test_all_seed${seed}" --kind infer -- \
    --checkpoint artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/training/train_gpu4_b16/checkpoints/best/best-000017439.ckpt \
    --split test --seed "$seed" --robots 0 1 2 --no-gt-comparison || break
done
```

每seed一个目录，可在不同GPU交互节点并行执行不同seed；同一步骤名只启动一次。
当前infer不支持恢复中断的目录，续跑需新步骤名；本模块不会据缺失预测计算正式指标。
所有三个目录完成后，在metrics环境运行：

```bash
conda activate mvwd-metrics
python scripts/run_mvwd_experiment.py run 20261013_fullscene_2frames_gpu4 \
  --name metrics_test_k3 --kind metrics_level1 -- \
  --predictions \
    artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/predictions/test_all_seed42 \
    artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/predictions/test_all_seed43 \
    artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/predictions/test_all_seed44 \
  --device cuda --batch-size 16
```

也支持同seed的多个不重叠预测目录，例如逐scene生成；跨目录重复query会报错。
所有输入需同checkpoint、renderer/机器人资产、推理参数、dataset release/split与分辨率；
每seed必须相同完整query集合。正式模式还会检查全部test episode及固定的2帧/3view选择。
未完成K=3、只抽部分scene或选取其他帧时只能跑`--preview`，避免best-of-K/样本偏置。

## 输出与复用

每个评估步骤在 `artifacts/mvwd/experiments/实验名/evaluations/步骤名/` 内保存：

- `metrics.json`：聚合数值、episode明细/CI、coverage、输入manifest hashes、模型后端来源。
- `protocol.yaml`：此次冻结的评估协议副本。
- `per_query.jsonl`：逐episode/seed/view/frame的PSNR和LPIPS。
- `per_pair.jsonl`：仅同一时刻、GT资格合格的MEt3R。
- `overlap.jsonl`：所有候选对的双向GT重叠率、有效像素数与资格，跨seed共享。
- `fid_*_statistics.npz`：全体特征的数量、均值和无偏协方差，按GT/各seed分别保存。

原prediction PNG/metadata不会改写，GT仅在评估时读取。终端日志和退出码由统一runner记录。
失败重试采用新步骤名。评估过程中会流式保存原始单位结果，完成后才写最终metrics.json；
当前不支持从半份评估结果自动恢复，不把半途结果当完整分数。

`mvwd_metrics`的核心API与Top2Pano网络解耦，供后续baseline/自己的方法使用：

```python
from mvwd_metrics import evaluate_level1
result = evaluate_level1(
    pred_images,       # [K,B,N,T,3,H,W]；K=1时也接受[B,N,T,3,H,W]
    gt_images,         # [B,N,T,3,H,W]，float RGB [0,1]
    camera_params,     # mapping (b,n,t) -> K/T_wc/已注册GT depth_z/near/far
    overlap_metadata,  # 或显式给出GT几何receipt；只由GT决定pair资格
    seeds=[42,43,44],
)
```

大规模数据用`Level1Evaluator.update_views/update_pair/finish`流式计算，避免一次加载所有图像。
Level1的T>1只代表多个独立同步时刻，不计算跨时间pair或GT-Warp。

## 已完成的真实GPU核验

在79775用户交互H100上运行，四个官方后端均已实际执行，未申请新资源。
Pomaria2预览结果见当前实验`evaluations/metrics_pomaria2_preview_retry3/metrics.json`；
Office预览见`evaluations/metrics_office_preview/metrics.json`，Grocery见`evaluations/metrics_grocery_preview/metrics.json`。每组都是1episode、9图、seed42。

| 预览 | PSNR | LPIPS | FID（仅9图诊断） | 生成MEt3R重建失败 |
| --- | ---: | ---: | ---: | ---: |
| Pomaria_2_int test | 11.5125 | 0.5728 | 232.8694 | 4/4 GT合格pairs |
| office_vendor_machine val | 11.7823 | 0.4196 | 161.7137 | 1/7 GT合格pairs |
| grocery_store_half_stocked val | 7.8871 | 0.7093 | 367.6624 | 2/5 GT合格pairs |

Pomaria/Office的GT参考对均有有效重建；Office有6个生成对成功得到官方MEt3R。Grocery未做GT参考。由于上述失效处理，
三组总体MEt3R均为undefined，不能用0冒充更好的一致性，也不发布仅成功子集的均值。
这不是完整test或K=3结果，FID只验证实现流程，不能据这些9图数值下主表结论。
失败尝试日志/步骤未删除；环境的完整pip freeze记录在同实验的evaluations/metrics_environment_freeze.txt。

最终验证：70项仓库测试通过，Python3.8语法兼容通过，独立指标环境pip check通过。
