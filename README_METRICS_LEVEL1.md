# MVWD Level 1 评估

当前 Top2Pano-Persp 是 Level 1：给定 BEV、机器人/相机状态，独立生成多个观察者图像。
抽取多个物理时刻时，每个时刻仍是独立的多视角条件，不因此成为 Level 3 视频模型。
本实现仅实现本任务所需的四个 Level 1 主指标；不混入 L2 edit/keep 或 L3 temporal 指标。

## 固定协议

当前默认协议是 `configs/mvwd_metrics_level1_gtwarp.yaml`：
**PSNR↑ / LPIPS↓ / FID↓ / MV-GTWarp-DINO↓**。它与模型训练配置分开。
原 `configs/mvwd_metrics_level1.yaml` 保留为官方MEt3R协议；旧结果不覆盖、不改名，两个指标不能混报。

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
| MV-GTWarp-DINO ↓ | GT几何对应位置上的DINO ViT-S/16＋FeatUp特征余弦误差；原生256×448完整FOV |
| 跨视角pair | 同一episode、同一物理帧、不同机器人；两个投影方向的像素均值再等权平均 |
| GT pair资格 | 双向可见表面占比的最小值 >0.1；GT深度注册到RGB、投影、目标深度一致，整个非零双线性采样邻域必须有效 |
| 可见性容差 | max(0.05m, 0.01×depth)，图像边界排除2px，并去掉深度不连续/invalid相邻像素 |
| 配对指标聚合 | episode/seed内部平均所有有效view或pair；seed均值；最后episode等权均值 |
| 95% CI | 固定seed20261015、2000次episode bootstrap；不把frame或seed当独立episode |

## MV-GTWarp-DINO定义与解释

`mv_gtwarp_dino` 是本benchmark定义的GT几何约束特征误差，**不是官方MEt3R**。
GT用于确定对应位置和可见性，不提供给模型生成。比较的是两张生成图的特征，不是逐图与GT的误差。
所有模型只需提交RGB；不使用3DGS自身深度、Top2Pano coarse depth、估计光流或生成图的重建结果。

对source像素p，用GT camera-forward depth反投影，经相机变换投影得到target连续坐标q。
使用像素中心(x+.5,y+.5)约定；RGB/depth各模态的实际相机位姿分别参与深度注册。
source与target均排除invalid、near/far之外、图像边界2px和深度不连续相邻像素。
**target中每个非零双线性权重对应的像素都必须通过可见性/深度检查**，避免跨遮挡边界混特征。
这比旧MEt3R的最近像素资格检查更严格，因此GT合格pair数可能不同；差别完全由GT决定。

冻结DINO ViT-S/16末层patch token(384维)，官方FeatUp dino16 `use_norm=True`恢复稠密特征。
输入保持256×448，不拉伸、不裁剪，RGB[0,1]采用ImageNet mean/std归一化；网络eval且无梯度。
在target q处双线性采样**未做L2归一化**的稠密特征，然后将source/target向量分别L2归一化。
单点误差为 `1 - cosine`，范围[0,2]；只裁剪余弦计算的浮点舍入越界。
方向内对全部GT有效像素平均，两个方向再等权平均（不按像素数加权）。
最后沿用episode/seed宏平均和episode-bootstrap CI。

同一时刻所有有效GT表面都参与，包括相互可见的机器人本体；不使用预测mask决定区域。
本实现仅同一时刻跨view，不实现跨时间机器人运动对应或Level3 temporal score。
未来静态环境的temporal GT-Warp可以复用几何/特征模块，但运动机器人需本体/link坐标对应。

没有GT合格pair的episode明确报告数量，不填0；若全部episode都没有，指标为null。
非有限或零范数的特征向量会让步骤失败，不根据生成结果删像素，不输出替代分数。
可加 `--gtwarp-gt-reference` 计算相同对应关系上的GT特征误差：总对照在`gtwarp_gt_reference`，
逐pair在`per_pair.jsonl`。对照仅用于诊断，不混入主分数、不相减、不做分母归一化。
GT特征误差因上下文、反光和特征感受野等可以非零，也不声称是严格数学下界。

**解释限制：**本指标测“在指定GT场景几何下的外观一致性”，并不证明任意生成三维场景自洽。
常量/模糊图可能得到较低一致性误差，必须与PSNR、LPIPS、FID结合；不要单列为生成质量。
这套v1实现已做基础对照，不等于已经完成大规模人类相关性或所有baseline的指标有效性验证。

原官方MEt3R仍可用：显式 `--protocol configs/mvwd_metrics_level1.yaml`；其GT几何仅选择候选对，
MEt3R自身仍推断生成图几何。空重建/非有限值逐pair记undefined；有失败时总MEt3R为null，
报告失败比例。不会将空重建的官方0误当成完美一致，也不只汇总成功子集。

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

原MEt3R协议中的MASt3R不是论文早期DUSt3R变体，采用官方当前默认的MASt3R配置；其他方法必须使用同一个
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
新GTWarp仅加载DINO/FeatUp，不加载MASt3R或PyTorch3D；当前独立环境已经具备所需依赖。

## 先评估已有preview

已有Pomaria2是1个episode、3个时刻×3个view、单seed42，不符合完整benchmark。
显式加`--preview`才接受这个范围；输出仍包含四指标，但不能作为完整test主表结果。

```bash
python scripts/run_mvwd_experiment.py run 20261013_fullscene_2frames_gpu4 \
  --name metrics_pomaria2_preview_new --kind metrics_level1 -- \
  --predictions artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/predictions/best_test_pomaria2 \
  --protocol configs/mvwd_metrics_level1_gtwarp.yaml \
  --preview --device cuda --gtwarp-gt-reference
```

只核验轻量指标时可显式`--preview --metrics psnr`，不加载torch/其他模型。
若缺依赖、权重或发生运行异常，命令返回失败并保留日志；不输出伪造或替代指标。
MEt3R重建无效属于可记录的metric undefined，评估流程可完成，但`all_requested_metrics_defined`
和`benchmark_ready`为false；completed不等于四个分数都可用。
若显式选择原MEt3R协议，可加`--met3r-gt-reference`额外核验同一GT相机对的官方重建，参考值仅用于诊断，不混入生成指标。

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
  --name metrics_test_k3_gtwarp --kind metrics_level1 -- \
  --protocol configs/mvwd_metrics_level1_gtwarp.yaml \
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
- `METRICS.md`：每次eval完成后自动生成的易读记录，包含总指标/CI、各seed、各场景、跨view覆盖及解释备注。
- `protocol.yaml`：此次冻结的评估协议副本。
- `per_query.jsonl`：逐episode/seed/view/frame的PSNR和LPIPS。
- `per_pair.jsonl`：同一时刻GT合格pair的MV-GTWarp-DINO、双向误差、像素数；选原协议时记录MEt3R。
- `overlap.jsonl`：所有候选对的双向GT重叠率、有效像素数与资格，跨seed共享。
- `fid_*_statistics.npz`：全体特征的数量、均值和无偏协方差，按GT/各seed分别保存。

已完成的旧评估也可以补导 `METRICS.md`，只读取已保存JSON/JSONL，不加载模型、不需要GPU，
原始 `metrics.json` 不改写。输出紧邻原指标，不新增reports目录；已有 `METRICS.md` 不覆盖。

```bash
conda activate mvwd-metrics
python scripts/run_mvwd_experiment.py run 20261013_fullscene_2frames_gpu4 \
  --name metrics_test_k3_readable_new --kind metrics_report -- \
  --evaluation artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/evaluations/metrics_test_k3_gtwarp
```

本次完整test评估已经补好，不需要再执行这个示例。其他评估换对应实验名和evaluation路径。

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
    # GTWarp必须提供camera_params，不能仅提供一个overlap比例。
    seeds=[42,43,44],
)
```

大规模数据用`Level1Evaluator.update_views/update_gtwarp_pair/finish`流式计算，避免一次加载所有图像。
Level1的T>1只代表多个独立同步时刻，不计算跨时间pair或temporal GT-Warp。

## 原MEt3R的真实GPU核验（保留历史记录）

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

## GTWarp真实GPU核验与对照

在用户交互H100作业79846内复用已有预测，未重新训练或生成：

- `evaluations/gtwarp_office_preview/`：Office 1episode/9图/seed42，MV-GTWarp-DINO=0.637552。
- `evaluations/gtwarp_pomaria2_preview/`：Pomaria2 1episode/9图/seed42，MV-GTWarp-DINO=0.841646。
- `evaluations/gtwarp_grocery_preview/`：Grocery 1episode/9图/seed42，MV-GTWarp-DINO=0.547154。
- `evaluations/gtwarp_val_controls/`：Office+Grocery validation的GT、错位、错配episode、重复视图、模糊、纯色对照。

上述路径均相对于当前实验`artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/`。
原preview的PSNR/LPIPS/FID保持一致；这些仍是小样本诊断，不是完整test K=3。

验证集11对的诊断pair均值：GT=0.435614，生成=0.604680，target平移32px=0.557867，
重复source视图=0.682049，模糊GT对=0.475959，纯灰对=0.086622；
错配其他episode target=0.779341（仅7对有独立donor，不能与11对均值直接作严格配对比较）。
纯灰低分是退化例子，不隐藏、不调整metric来强制它变差；质量指标必须一起报告。
所有逐pair结果保存在`controls.jsonl`，`validation.json`明确标为diagnostic、pair均值而非benchmark宏平均。

需要复验时使用新步骤名，两个输入必须来自同split、同checkpoint和生成协议：

```bash
python scripts/run_mvwd_experiment.py run 20261013_fullscene_2frames_gpu4 \
  --name gtwarp_val_controls_new --kind gtwarp_validate -- \
  --predictions \
    artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/predictions/best_val_office \
    artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/predictions/best_val_grocery \
  --device cuda --max-pairs 12
```

该命令只做少量诊断，不替代完整test评估。源码、协议、已加载权重和模型状态校验随结果记录。

本轮最终核验：79项全仓库测试通过、9项最终GTWarp针对性测试通过、Python3.8语法通过。
两个val episode的最终FP32宏平均=0.592353；汇总见当前实验evaluations/gtwarp_implementation_validation.json。
