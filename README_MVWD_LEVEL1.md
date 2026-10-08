# Top2Pano-Persp：MVWD Level 1

后续实验统一使用 [实验目录与日志规范](README_EXPERIMENTS.md)；每个实验集中保存配置、命令、日志、checkpoint、预测和报告。本文中的直接脚本调用说明底层接口，旧 artifacts 路径仅供历史追溯。

在本仓库内独立读取 `MultiViewWorldDataset_v1` 的 JSONL/JSON/NPZ，**不导入
`mvwd.py`、consumer_tools 或 simulator/producer 包**。不重新采集或修改正式数据，不重划 splits。机器人资产的投影在 baseline 内完成。

输入是无机器人 **before environment BEV RGB**、BEV 标定、目标实际采集相机 K/T、
**当前物理帧所有机器人的 base-to-world 位姿、身份、相机高度和共享带纹理机器人资产**。
训练监督为 **before perspective RGB + depth**，segmentation 由输入 RGB 经 SAM
生成，绝不读取 GT semantic/height/occupancy。每个机器人/帧独立生成，不加入多视角
attention、联合 diffusion 或跨视角/时间损失。默认 BEV 512×512，目标 H×W=256×448。

## 以实际仓库代码为准

按照用户要求，保留原代码中的网络、损失计算和 `CombinedModel` 训练调用；不根据论文
重写损失或 occupancy 学习过程。原 Gibson/Matterport 分支继续保留。

特别是，以下行为被**明确保留**：

- `main.py` 的 density `shared_step` 修改 batch，但其返回的 density loss 没有加入最终 loss。
- occupancy 从 BEV 编码得到的 `x_start` 解码；没有改成从 density denoiser 预测值解码。
- `decode_first_stage` 的 `torch.no_grad()` 和 first-stage 参数冻结保持原样。
- refinement 的 clean latent 从 `x_start` 调用 `predict_start_from_noise`，而非从 `x_noisy`。
- color histogram 使用不可微的 `torch.histc`；原有 `loss_type` 修改、归一化和深度损失保持原样。
- optimizer 仍为原 `main.py` 的 Adam 和两个 stage 的全部参数列表。

因此，**原 CombinedModel 流程下 density stage 不获得梯度，解码后的 depth/color 辅助
损失也不提供新的解码梯度**。本适配保留这个代码事实，不能把它描述成已修复的、按论文
重新实现的端到端 OccRecon。测试会核对这一点。若未来决定修复，应作为另一个明确命名的
variant，而不是悄悄改变这个版本。

必要修改仅涉及：

1. baseline 自己的 raw adapter 与单视角 query 索引。
2. 标定后的 pinhole rays 和 world→BEV→voxel 查询，去除当前相机不需要的 pano 旋转/翻转。
3. BEV 等比例 padding/resize 与标定更新；目标等比例 resize 与 K 更新。
4. camera-forward GT depth 的相机注册，以及渲染距离到 forward depth 的转换。
5. 非方形输出的独立推理/保存入口，避免原 logger 强制转回 512×512。
6. 原环境密度与已知机器人表面的 RGB/depth 合成；refinement 继续使用原有条件通道。
7. 三处设备处理：decoder depth head 随模型移动、CLIP tokens 跟随文本模型、DDIM buffers
   跟随模型设备。这些不改变网络结构或损失。

透视 renderer 保留原密度 min/max→×10、RGB 输入范围、volumetric alpha compositing、
两层 solid floor、RGB-dark wall heuristic 和 RGB 短采样范围 0.8。有效 BEV padding 不作为墙。
高度仍默认 3 m，但改为显式参数；不再按绝对世界 z 加减 3/6 m。深度范围使用相机 near/far，
保留每张图的相对深度归一化。数值除法增加零范围保护。

颜色直方图保留原 `[0,1]` 范围、跨 batch 聚合和不可微 `torch.histc`，仅对计数为零的
通道返回零直方图，避免 `hist / 0` 产生 NaN；非空通道结果与原代码一致。这项数值
修复不重新缩放或截断颜色，不改变损失权重和网络，也不会为辅助项增加梯度。
旧 checkpoint 的结构兼容，参数检查通过后可继续使用；旧日志中的 NaN 不会被改写。

公开权重、初始化和实际更新参数已逐项核验，见 [权重与训练流程核验](README_UPSTREAM_AUDIT.md)。
作者两个发布 checkpoint 未发现不同于原 SD 的 coarse VAE 权重；当前初始化的复制规则及
完整文件 SHA256 均通过检查。

## 机器人如何进入生成链路

默认配置的 `data.robot_rendering` 和 `renderer.robot_rendering` 均为 `true`。
资产默认读取 `<data.root>/robot_assets/MVWD_RobotAssets_v1`，也可通过 `--robot-assets`
指定新位置。读取和渲染实现均在本 baseline 的 `adapters/robot_assets.py` 中，不导入
资产包的 loader 或 Isaac Sim。启动时核对 episode 的机器人 template fingerprint；
实际使用的元数据和 GLB 核对 manifest 中的大小及 SHA256。

每个 query 读取 `trajectories.npz` 中**所有机器人在同一帧的计划 base-to-world 位姿**，
再读取各机器人相机高度选择 mast 几何。目标相机仍使用 observations 中的实际 K/T。
即使只输出 robot 0 的视角，也保留 robot 1/2 的状态；不会使用未来帧作为条件。测量高度
允许 1 mm 以内偏差并映射到 0.8/1.0/1.2/1.4 m 的已发布资产规格。轮子等其他关节使用
资产发布的静止姿态。计划轨迹和实际采集位姿的小偏差仍可能带来投影误差。

使用 GLB 的完整三角形、部件变换、嵌入纹理 UV、PBR base-color factor 和身份颜色，
构建 CPU Embree BVH。每个高度的 BVH 在进程内复用，机器人移动只改变射线坐标变换。
当前每个高度约 314 万三角形，首次构建会增加启动时间及 CPU 内存占用；默认 workers=0
避免各 worker 重复持有资产。不需要重新运行 SAM，环境 BEV/SAM 缓存保持共用。

同一套目标相机射线先找最近的机器人表面，再积分该表面之前的**预测环境密度**：
机器人互相遮挡由最近表面决定，环境遮挡由预测密度的 transmittance 决定。机器人表面
截断后方环境积分，向 coarse RGB 和米制 depth 加入相应颜色与深度。原环境 RGB 的
0.8 短采样范围保留，机器人使用完整 camera near/far 范围。观察者自己的机体若出现在
相机视野中，也正常渲染。推理不读取 GT depth、GT mask、GT visibility 或场景 mesh。

这是**不透明表面的基础颜色引导**，没有重现 Isaac RTX 的照明、阴影、金属反射或 MDL
材质效果。最终 RGB 继续由原 Top2Pano 的单视角 refinement 生成；没有增加机器人专用
网络、mask loss、时序 loss 或最终图像硬贴图。coarse 中存在机器人并不保证 diffusion
输出一定保留其轮廓与身份，正式实验需要检查并评价最终结果。预测环境错误也会造成
错误遮挡；原 density stage 不获得梯度的代码行为仍保留。

带机器人版本默认写入 `artifacts/mvwd/runs/level1_robots/`，旧 `level1/` 结果保留。
`models/control_sd21_ini.ckpt` 可直接复用。旧无机器人训练 checkpoint 不允许作为此版本的
resume/inference checkpoint；加载时核对机器人条件开关和资产 manifest SHA256。旧实验
若需复现，使用开关均为 `false` 的独立配置及原输出目录。

## 固定 seed 的机器人条件诊断

`scripts/diagnose_robot_conditions.py` 独立检查训练后模型的条件响应：固定 BEV、目标相机、
geometry/query seed，比较原条件、移除其他机器人、沿相机水平右方向移动其他机器人、交换
其他机器人的身份颜色。观察者自身的位姿、机体和相机保持原样；不修改训练或权重。

例如，对已有带机器人 checkpoint 执行：

```bash
python scripts/diagnose_robot_conditions.py \
  --config artifacts/mvwd/night_20261008/taskA/config.yaml \
  --checkpoint artifacts/mvwd/night_20261008/taskA/run/checkpoints/last.ckpt \
  --scene Rs_int --max-episodes 2 --robots 1 2 --frames 10 \
  --guidances 1 9 --shift-m 0.75 \
  --output artifacts/mvwd/condition_diagnostics/taskA/interventions_v2
```

输出目录必须不存在。保存原/反事实 coarse、最终 RGB、`sensitivity.json` 和完成标记。
诊断不读取目标 RGB/depth，不对没有对应 GT 的反事实计算 PSNR/SSIM。像素变化只能证明
数值条件敏感性，不能证明机器人数量、身份或运动正确；还需查看配对图。原 renderer 的
全图 min/max 归一化可能让局部机器人变化影响全图条件，移动也未经过场景碰撞验证。
本入口不用于正式 benchmark 的预测或选模。

## 文件与环境

```text
adapters/mvwd_raw.py            pinned split、JSON/NPZ 读取
adapters/mvwd_level1.py         单视角 sample、resize、标定、SAM 条件
adapters/camera_geometry.py     透视射线、环境与机器人 alpha compositing
adapters/robot_assets.py        资产校验、纹理 UV、CPU BVH 与机器人表面查询
mvwd_runtime.py                 原 batch 字段转换、depth 注册、实验元数据
mvwd_model.py                   原 CombinedModel 调用、独立推理
configs/mvwd_level1.yaml        数据、模型、训练与推理配置
scripts/inspect_mvwd.py         splits 与 BEV 轨迹可视化
scripts/inspect_robot_rendering.py  真实机器人资产与相机投影检查
scripts/prepare_mvwd.py         SAM 条件和可选 npy/memmap 缓存
train_mvwd.py / infer_mvwd.py   训练与推理
```

adapter 已兼容当前 Python 3.8 环境，使用 NumPy、Pillow ≥9.1、PyYAML；模型保留原
Top2Pano 实现。目前已验证的 `top2pano` 环境为 Python 3.8.20、PyTorch 2.0.1、
TorchVision 0.15.2、CUDA build 11.8 和 Lightning 1.5.0，已有环境无需重建；本次已安装 `trimesh==4.5.3` 和 `embreex==2.17.7`。
其他机器或旧环境补充机器人渲染依赖：

```bash
python -m pip install -r requirements_robot_rendering.txt
```

`environment_mvwd.yaml` 已更新为当前已验证环境的完整快照；需要重建时使用该文件。
原 `environment.yaml` 保留上游历史依赖并已标注用途。HOME 同步保留 YAML、Conda explicit
和 pip freeze 快照，位于 `~/.config/multiview/environments/`。

```bash
source ~/.config/multiview/storage.sh
conda activate top2pano
cd ~/multiview_baselines/top2pano
```

HOME 仓库入口是 workspace 实际仓库的软链接，可继续编辑和运行。完整存储、环境及
路径说明见 `~/.config/multiview/STORAGE_USAGE.md`。

配置中的相对路径均以解析后的实际仓库为基准。迁移后默认 dataset root 为
`/hfs2/work/workspace/scratch/ka_wq8392-multiview/datasets/MultiViewWorldDataset_v1`；
也可用各命令的 `--root` 指定。
不会依赖启动时的工作目录来定位 payload。

## 集群调试：共用一个交互式 H100 作业

数据检查、权重初始化、SAM 小子集预处理、I/O 缓存、小规模训练、推理和测试，均在
同一个交互式计算节点中直接执行下面的 Python 命令。先申请一次资源，再按步骤调试。

在 HoreKa-2 登录节点上创建 tmux 会话，并申请资源：

```bash
tmux new -s top2pano-h100

salloc \
  --job-name=top2pano-debug \
  --partition=gpu-h100 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=2-00:00:00
```

等待资源分配成功后，进入计算节点：

```bash
srun --gres=gpu:1 --pty bash -l
hostname
nvidia-smi
conda activate top2pano
cd /hfs2/data/home/ka_anthropomatik/ka_wq8392/multiview_baselines/top2pano
mkdir -p artifacts/mvwd/debug_logs
set -o pipefail
```

`gpu-h100` 最长允许 48 小时，`dev-gpu-h100` 最长 1 小时；这里使用普通分区。
资源仍需等待调度，48 小时从分配成功后开始计时。进入 GPU shell 时保留 `--gres=gpu:1`。
参见 [HoreKa-2 分区及交互作业说明](https://docs.nhr.kit.edu/usage/slurm/)。

先验证当前环境的 CUDA 前向和反向计算，再开始 SAM 或训练：

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda", requires_grad=True)
(x @ x).square().mean().backward()
torch.cuda.synchronize()
print("CUDA forward/backward passed")
PY
```

后续命令都在这个 shell 内执行。需要保存终端日志时，例如：

```bash
python -u scripts/inspect_mvwd.py --scene Rs_int 2>&1 \
  | tee artifacts/mvwd/debug_logs/inspect.log
```

按 `Ctrl-b` 再按 `d` 可暂时离开 tmux；重新连接到同一台登录节点后，执行
`tmux attach -t top2pano-h100` 恢复会话。tmux 保留 SSH 断开后的会话，作业到时仍会结束。
调试结束后，先 `exit` 离开计算节点 shell，再 `exit` 退出 salloc shell，释放资源。

## 第一步：检查真实数据

在上述交互式计算节点内执行；此步只用 CPU，无需模型权重或 SAM 缓存：

```bash
python scripts/inspect_mvwd.py --scene Rs_int
```

结果在 `artifacts/mvwd/inspection/`，包括 `bev_trajectories.png`、三个机器人第一帧
RGB、`bev_valid.png` 和 `inspection.json`。该入口检查 train/val/test 场景、episode 和
configuration 无交集，并检查新 BEV 标定的坐标往返一致性。

先查看 BEV 轨迹和目标 RGB，再准备 SAM；完整 sample 检查见下一步的 `--load-query`。

## 第二步：准备 SAM 条件

### 权重下载与 SD 2.1 初始化

在已激活 `top2pano` 的交互式计算节点内执行。SAM 使用官方的 SAM 1 ViT-H：

```bash
mkdir -p models/sam
wget -c -O models/sam/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

来源：[SAM 官方 checkpoint 清单](https://github.com/facebookresearch/segment-anything#model-checkpoints)。
`--sam-model vit_h` 必须与这个文件匹配。

`control_sd21_ini.ckpt` 按 [ControlNet 官方 SD 2.1 初始化说明](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md)
由 **SD 2.1 512-base** 生成。原 Stability AI 地址在本次检查返回 401；以下使用明确标注的
[sd2-community 存档文件](https://huggingface.co/sd2-community/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)，
并核对存档页公布的 SHA256：

```bash
wget -c -O models/v2-1_512-ema-pruned.ckpt \
  https://huggingface.co/sd2-community/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt

printf '%s  %s\n' \
  88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36 \
  models/v2-1_512-ema-pruned.ckpt | sha256sum -c -

python scripts/init_control_sd21.py \
  --input models/v2-1_512-ema-pruned.ckpt \
  --output models/control_sd21_ini.ckpt
```

本地工具遵循 [官方 `tool_add_control_sd21.py`](https://github.com/lllyasviel/ControlNet/blob/main/tool_add_control_sd21.py)
的参数复制规则：将对应 SD UNet 参数复制到 ControlNet，其他可匹配参数同名复制，新层保留
本仓库模型的初始化。它增加 CPU 加载、shape 检查和来源/输出 SHA256 receipt，不改变模型。
`first_stage_model.decoder.depth_map.*` 等 Top2Pano 新层的初始化会在报告中列出。

初始化在 CPU 上完成，输入模型约 5.21 GB，生成文件和 OpenCLIP 缓存也占用数 GB。
首次模型创建可能自动下载原 OpenCLIP 资源，建议在能联网且 CPU 内存充足的节点执行。
成功后得到 `models/control_sd21_ini.ckpt` 和 `models/control_sd21_ini.json`，训练配置无需改动。

先提供下载好的 SAM checkpoint；配置的初始化模型文件也需单独提供。
本实现不伪造 segmentation、不使用 GT semantic，也不静默降级为全零条件。

```bash
python scripts/prepare_mvwd.py --mode sam \
  --splits train --scene Rs_int --max-episodes 2 \
  --sam-model vit_h --sam-checkpoint models/sam/sam_vit_h_4b8939.pth
```

SAM 小子集准备完成后，在同一个 shell 检查完整 adapter sample 和 segmentation：

```bash
python scripts/inspect_mvwd.py --scene Rs_int --load-query
```

`--load-query` 只读取已有 SAM 缓存，不运行 SAM 网络；该检查不需要 GPU。

这是原方法的 class-free segmentation conditioning。仓库没有提供原 `_seg.png` 的生成
脚本，因此此处补充 `SamAutomaticMaskGenerator`：在 letterbox 后的输入 RGB 上生成 masks，
按面积从大到小覆盖，固定 seed 着色为 3-channel RGB。此预处理细节不是对原 `_seg.png`
逐像素的复现，实验中应记录它。checkpoint SHA256、参数、着色规则均写入 cache receipt。

缓存按 `(configuration_path, floor_index, bev_size)` 去重。使用 `configuration_path`
而非只有 `configuration_id`，避免将不同来源 payload 合并。SAM 配方变化时需要使用不同
`segmentation_root`，脚本不会混合配方。全集准备命令：

```bash
python scripts/prepare_mvwd.py --mode sam \
  --sam-model vit_h --sam-checkpoint /path/to/sam_vit_h.pth
```

## 第三步：可选的训练 I/O 缓存

压缩 NPZ 读取一帧也可能解压整个 60-frame 数组。先在小子集验证，之后再按预算准备：

```bash
python scripts/prepare_mvwd.py --mode views \
  --splits train --scene Rs_int --max-episodes 2
```

缓存只提取 RGB 和 depth，不改正式 raw data；npy 使用 mmap 逐帧访问。每个机器人原 RGB
解压约 83 MB，全集派生缓存会占用大量磁盘，所以**不要在未评估磁盘容量前直接缓存全集**。
未缓存的样本自动使用 raw NPZ，进程内只保留一个机器人视频包。默认 workers=0。

## 第四步：检查机器人投影，再训练

先运行机器人几何检查，不分配 diffusion 网络；需要已经准备好的小子集 SAM 条件：

```bash
python scripts/inspect_robot_rendering.py --scene Rs_int \
  --max-episodes 1 --robots 0 1 2 --frames 0 10 20
```

输出位于 `artifacts/mvwd/robot_inspection/`：`*.robot_rgb.png` 是黑色背景上的机器人
基础颜色，`*.robot_id.png` 按身份着色，`*.robots.npz` 包含机器人表面深度和身份。
**该诊断仅检查机器人投影，不包含环境遮挡**；黑色背景不代表生成的室内图像。
可加 `--with-targets` 保存 GT RGB 供人工比较，这些图不参与渲染。正常训练与推理的
环境遮挡由 `PerspectiveRenderer` 的预测密度处理。检查报告明确记录是否打开过 GT。


准备 `models/control_sd21_ini.ckpt`，或修改配置里的 `model.init_checkpoint`；该文件像原
`main.py` 一样分别初始化两个 stage。CLIP 按原 model config 使用其预训练资源。
在同一个交互式 GPU shell 先运行小样本：

```bash
python train_mvwd.py --scene Rs_int --max-episodes 2 \
  --frame-stride 10 --max-steps 20 --no-validation
```

默认配置只训练 1 epoch，是开发默认值。正式实验请明确修改 epoch/step budget、batch 和
gradient accumulation，并先准备 val SAM 条件。全量入口：

```bash
python train_mvwd.py
```

`--resume /path/to/last.ckpt` 恢复训练。实验目录保存配置、release、split SHA256、样本选择、
SAM 配方、机器人资产 provenance 和初始化 checkpoint SHA256，以及 split/dataset 元数据
副本。输入不包含目标 RGB 视频或 depth 视频。

## 第五步：独立推理

先在已有 SAM 缓存的 train 小子集做链路检查；这不是正式测试指标。在同一个交互式 GPU
shell 内执行：

```bash
python infer_mvwd.py \
  --checkpoint artifacts/mvwd/runs/level1_robots/checkpoints/last.ckpt \
  --split train --scene Rs_int --max-episodes 1 --robots 0 1 2 --frames 0 10 20 \
  --output artifacts/mvwd/predictions/robots_smoke
```

正式测试时改为 `--split test`、移除 train 的 `--scene Rs_int`，并提前准备对应 SAM 缓存。
推理仅读取 before BEV、标定、SAM 条件、实际目标相机和当前帧所有机器人状态及资产，
不打开目标 RGB/depth NPZ。
同一 BEV 共用一次 occupancy，appearance refinement 独立进行。默认每个 query 的 diffusion
seed 由 base seed/release/episode/robot/frame 稳定确定；occupancy seed 由 BEV key 稳定确定。

保存 `episode/robot/frame.png`、coarse RGB/米制 forward depth 和 `predictions.jsonl`。
coarse NPZ 还保存 `robot_id`、`robot_weight`、`robot_depth_z`，用于检查几何条件。
`robot_id` 是最近机器人表面身份，包含被环境挡住的表面；`robot_weight` 才是预测环境
透射后该表面的贡献，二者都不是 GT 可见性标签。输出保留机器人和时间对应关系，
接入统一 Level 1 evaluator 时仍评价最终生成图像，本版本不新增评价协议。

当前数据还包含 garden 等场景，代码不会自动过滤。3 m 高度和 RGB-dark wall prior 沿用
原实现；应通过可视化和分组结果评估它们在不同场景的适用性，而不是使用 GT 几何修补。

## 验证范围

测试也在上述交互式计算节点内直接运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/top2pano-level1/bin/python -m pytest -q -p no:cacheprovider tests
```

测试覆盖标定/像素中心/FoV、实际 modality pose、GT-free adapter、depth 注册、scene
平移与楼层偏移不变性、非方形 renderer、分块/批处理/梯度，以及真实 ControlLDM loss、
Lightning 训练/验证/checkpoint 与 DDIM 推理的 CPU 小模型链路。机器人测试还覆盖
全体状态读取、纹理 UV 与材质因子、身份颜色、移动投影、近远裁剪、机器人之间与预测
环境的遮挡、坐标整体平移，以及错误资产和旧 checkpoint 拒绝加载。小模型只替换测试用文本
fixture 并缩减模型宽度，不下载预训练模型；它不代表预训练大模型的效果或 GPU 内存验证。

本机另建的 `.venvs/top2pano-level1` 是 CPU 验证环境，不是已准备好权重的 GPU 正式训练环境。
完整训练仍需 GPU、ControlNet/Top2Pano 初始化权重、SAM 权重与相应条件缓存。

本次已检查真实 Rs_int 的 3 个视角 × 3 帧机器人投影，并在 Python 3.8 中验证真实资产与
完整透视 renderer 的零密度数值合成。结果见 `artifacts/mvwd/robot_inspection/`。这些检查
和 CPU 小模型测试证明实现链路，带机器人版本的完整预训练模型 H100 训练及最终生成
质量仍需执行上面的小规模训练/推理命令验证。

若训练日志出现 NaN，可先在内存充足的计算节点检查全部模型和优化器状态：

```bash
python scripts/check_mvwd_checkpoint.py \
  --checkpoint artifacts/mvwd/runs/level1_robots_overfit/checkpoints/last.ckpt \
  --output artifacts/mvwd/runs/level1_robots_overfit/checkpoint_finiteness.json
```

该检查在 CPU 上完整加载 checkpoint，当前文件约 19 GiB；它不构建网络或修改权重，
也不证明模型收敛或生成质量。已有 500-step 训练日志中的颜色项 NaN 已复现为空直方图
除零；对应权重和 Adam 状态均有限，修复后的实际模型检查见同目录的
`loss_numerics_verification.json`。
