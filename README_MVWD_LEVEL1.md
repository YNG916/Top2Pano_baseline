# Top2Pano-Persp：MVWD Level 1

在本仓库内独立读取 `MultiViewWorldDataset_v1` 的 JSONL/JSON/NPZ，**不导入
`mvwd.py`、consumer_tools 或 simulator/producer 包**。不重新渲染数据，不重划 splits。

输入是无机器人 **before environment BEV RGB**、BEV 标定和目标实际采集相机 K/T。
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
6. 三处设备处理：decoder depth head 随模型移动、CLIP tokens 跟随文本模型、DDIM buffers
   跟随模型设备。这些不改变网络结构或损失。

透视 renderer 保留原密度 min/max→×10、RGB 输入范围、volumetric alpha compositing、
两层 solid floor、RGB-dark wall heuristic 和 RGB 短采样范围 0.8。有效 BEV padding 不作为墙。
高度仍默认 3 m，但改为显式参数；不再按绝对世界 z 加减 3/6 m。深度范围使用相机 near/far，
保留每张图的相对深度归一化。数值除法增加零范围保护。

## 文件与环境

```text
adapters/mvwd_raw.py            pinned split、JSON/NPZ 读取
adapters/mvwd_level1.py         单视角 sample、resize、标定、SAM 条件
adapters/camera_geometry.py     透视射线、分块 alpha compositing
mvwd_runtime.py                 原 batch 字段转换、depth 注册、实验元数据
mvwd_model.py                   原 CombinedModel 调用、独立推理
configs/mvwd_level1.yaml        数据、模型、训练与推理配置
scripts/inspect_mvwd.py         splits 与 BEV 轨迹可视化
scripts/prepare_mvwd.py         SAM 条件和可选 npy/memmap 缓存
train_mvwd.py / infer_mvwd.py   训练与推理
```

adapter 已兼容当前 Python 3.8 环境，使用 NumPy、Pillow ≥9.1、PyYAML；模型保留原
Top2Pano 实现。目前已验证的 `top2pano` 环境为 Python 3.8.20、PyTorch 2.0.1、
TorchVision 0.15.2、CUDA build 11.8 和 Lightning 1.5.0，已有环境无需重建。
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

## 第四步：训练

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
SAM 配方和初始化 checkpoint SHA256，以及 split/dataset 元数据副本。不会引入 RGB 视频
或几何视频作为输入条件。

## 第五步：独立推理

准备好选定 split/episode 对应的 SAM 缓存后，在同一个交互式 GPU shell 内执行：

```bash
python infer_mvwd.py \
  --checkpoint artifacts/mvwd/runs/level1/checkpoints/last.ckpt \
  --split test --max-episodes 1 --robots 0 1 2 --frames 0 10 20 \
  --output artifacts/mvwd/predictions
```

推理仅读取 before BEV、标定、SAM 条件和实际目标相机，不打开目标 RGB/depth NPZ。
同一 BEV 共用一次 occupancy，appearance refinement 独立进行。默认每个 query 的 diffusion
seed 由 base seed/release/episode/robot/frame 稳定确定；occupancy seed 由 BEV key 稳定确定。

保存 `episode/robot/frame.png`、coarse RGB/米制 forward depth 和 `predictions.jsonl`。
输出保留机器人和时间对应关系，方便接入统一 Level 1 evaluator；本版本不擅自定义机器人
mask 或新增评价协议。输入中若不包含机器人，它们出现于 GT 的处理应由 benchmark 统一决定。

当前数据还包含 garden 等场景，代码不会自动过滤。3 m 高度和 RGB-dark wall prior 沿用
原实现；应通过可视化和分组结果评估它们在不同场景的适用性，而不是使用 GT 几何修补。

## 验证范围

测试也在上述交互式计算节点内直接运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/top2pano-level1/bin/python -m pytest -q -p no:cacheprovider tests
```

测试覆盖标定/像素中心/FoV、实际 modality pose、GT-free adapter、depth 注册、scene
平移与楼层偏移不变性、非方形 renderer、分块/批处理/梯度，以及真实 ControlLDM loss、
Lightning 训练/验证/checkpoint 与 DDIM 推理的 CPU 小模型链路。小模型只替换测试用文本
fixture 并缩减模型宽度，不下载预训练模型；它不代表预训练大模型的效果或 GPU 内存验证。

本机另建的 `.venvs/top2pano-level1` 是 CPU 验证环境，不是已准备好权重的 GPU 正式训练环境。
完整训练仍需 GPU、ControlNet/Top2Pano 初始化权重、SAM 权重与相应条件缓存。
