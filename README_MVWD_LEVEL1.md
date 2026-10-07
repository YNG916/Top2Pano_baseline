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

adapter 需要 Python ≥3.9、NumPy、Pillow ≥9.1、PyYAML；模型使用原 Top2Pano dependencies。
新增 `environment_mvwd.yaml` 将 Python 调整为 3.10，并提供这条入口需要的依赖。
原 `environment.yaml` 不变。所有命令在本仓库目录执行：

```bash
conda env create -f environment_mvwd.yaml
conda activate top2pano-mvwd
```

配置中的相对路径均以本仓库为基准。默认 dataset root 指向现有
`../../multiview_ws/datasets/MultiViewWorldDataset_v1`；也可用各命令的 `--root` 指定。
不会依赖启动时的工作目录来定位 payload。

## 第一步：检查真实数据

无需模型权重或 SAM 缓存：

```bash
python scripts/inspect_mvwd.py --scene Rs_int
```

结果在 `artifacts/mvwd/inspection/`，包括 `bev_trajectories.png`、三个机器人第一帧
RGB、`bev_valid.png` 和 `inspection.json`。该入口检查 train/val/test 场景、episode 和
configuration 无交集，并检查新 BEV 标定的坐标往返一致性。

## 第二步：准备 SAM 条件

先提供下载好的 SAM checkpoint；配置的初始化模型文件也需单独提供。
本实现不伪造 segmentation、不使用 GT semantic，也不静默降级为全零条件。

```bash
python scripts/prepare_mvwd.py --mode sam \
  --splits train --scene Rs_int --max-episodes 2 \
  --sam-model vit_h --sam-checkpoint /path/to/sam_vit_h.pth
```

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
先运行小样本：

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

```bash
python -m pytest -q tests
```

测试覆盖标定/像素中心/FoV、实际 modality pose、GT-free adapter、depth 注册、scene
平移与楼层偏移不变性、非方形 renderer、分块/批处理/梯度，以及真实 ControlLDM loss、
Lightning 训练/验证/checkpoint 与 DDIM 推理的 CPU 小模型链路。小模型只替换测试用文本
fixture 并缩减模型宽度，不下载预训练模型；它不代表预训练大模型的效果或 GPU 内存验证。

本机另建的 `.venvs/top2pano-level1` 是 CPU 验证环境，不是已准备好权重的 GPU 正式训练环境。
完整训练仍需 GPU、ControlNet/Top2Pano 初始化权重、SAM 权重与相应条件缓存。
