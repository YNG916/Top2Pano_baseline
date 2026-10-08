# Top2Pano 公开权重、初始化与训练流程核验

核验日期：2026-10-08。范围为已发布代码和 checkpoint；本次不更改模型、loss、
冻结策略或 optimizer，不实现新的几何训练 variant。

**结论：当前 SD 2.1 → ControlNet 初始化正确，没有发现遗漏的已训练 coarse 几何权重。
当前流程实际更新 refinement；扩大训练预算不会使 coarse 几何随训练改善。**

## 证据与检查范围

作者代码固定在 commit `574295abe22e083965879a4ea51c6370a154726e`。
[公开训练入口](https://github.com/zhangzitong1312/top2pano/blob/574295abe22e083965879a4ea51c6370a154726e/main.py)
的 SHA256 与本地 `main.py` 完全一致，`test.py` 也完全一致。密度 loss 函数唯一差异
是 MVWD 透视分支跳过原全景图的旋转/翻转；去除这项已记录的 adapter 改动后 AST 相同。
`decode_first_stage` 的 no-grad 和 first-stage 冻结代码均与公开版本相同。

检查了[作者权重目录](https://huggingface.co/freeA1/top2pano/tree/main)中的两个 checkpoint：

| 权重 | 保存 epoch / step | density 与当前初始化匹配的张量 | density VAE 与原 SD 匹配的张量 |
| --- | --- | --- | --- |
| `gibson.ckpt` | 100 / 179300 | 1567 / 1583 | 248 / 248 原有 VAE 张量 |
| `matterport.ckpt` | 100 / 205900 | 1567 / 1583 | 248 / 248 原有 VAE 张量 |

每个 checkpoint 大小为 20,372,688,247 bytes，包含 3,166 个模型 state 张量和 728 个
Adam 参数状态。它们没有保存 `hyper_parameters`，因此不能由文件确定完整训练配置或数据划分。

两个作者 checkpoint 中，density 分支的 **1,538 个 SD 来源张量全部匹配当前初始化**。
剩余差异的 16 个张量均属于 SD 中不存在的新层：14 个 hint-block 权重/bias，以及 2 个
新增 depth-head 权重/bias。这些差异本身不能证明它们接受过训练。coarse 直接解码 BEV
编码 latent；该路径不使用 hint block，也不使用 decoder 的 RGB/depth 输出 head。
因此这些差异不提供另一套可用的 coarse 几何。

比对方法是逐个完整 tensor storage 的 dtype、shape、stride、size 和 ZIP CRC32；每个作者
文件另有 10 个代表性 tensor 的 SHA256 比对，涵盖 density/render 的 decoder、post-quant、
UNet 输入和 ControlNet zero-conv。CRC32 是全量筛查，不是每个张量的密码学证明。
HTTP range 读取每个文件约 2.42 MiB；没有下载两份约 19 GiB 的完整 checkpoint。
固定 Hugging Face revision、LFS SHA256、全部差异名和 anchor SHA256 记录在报告中。

## 当前初始化

实际检查了 `models/v2-1_512-ema-pruned.ckpt` 和 `models/control_sd21_ini.ckpt`：

- SD 输入文件和 ControlNet 输出文件的完整 SHA256 均与初始化 receipt 一致。
- 1,538 个复制张量逐项匹配；`control_* ← model.diffusion_*` 映射没有错配。
- 45 个新初始化张量的名单与 receipt 一致。
- 新 depth head 没有 SD 预训练权重，当前保留仓库原始初始化；这符合原初始化规则。

这里的 `control_sd21_ini.ckpt` 是 SD/ControlNet 初始化，不能描述成作者训练好的 Top2Pano。

## 当前训练确实更新了什么

本地 `artifacts/mvwd/runs/level1_robots_overfit/checkpoints/last.ckpt` 为 step 500：

| 分支 | 相对初始化发生变化的张量 | 说明 |
| --- | --- | --- |
| density | 0 / 1583 | coarse 来源保持原样 |
| refinement | 728 / 1583 | ControlNet 和 UNet output 部分接受更新 |
| refinement VAE | 0 / 250 | 原 VAE 与新增 depth head 均冻结 |

当前及作者发布的 `CombinedModel` 丢弃 density `shared_step` 的返回 loss；occupancy 解码
使用编码 BEV 的 `x_start`，不是 density denoiser 的预测值。first-stage 冻结且解码有
`torch.no_grad()`。refinement 的 depth/color 辅助项虽然计入返回 loss，但没有相应解码梯度。
因此日志中出现这些 loss 数值，不能作为对应模块已被训练的证据。

Combined optimizer 使用两分支全部参数；不能仅由 `sd_locked: true` 宣称只训练 ControlNet。
`ControlledUnetModel.forward` 对 input/middle 使用 no-grad，output 部分仍可更新。
作者两个 checkpoint 的 refinement 有 730 个张量区别于本地初始化，其中包含 2 个新增
随机 depth-head 张量的初始差异；其 728 个 Adam 参数状态与上述公开流程相符。

本次重新执行 `tests/test_original_model_flow.py`：**3 passed**。实际小模型覆盖无机器人和
带机器人两种训练/反向、Lightning 单步/验证/checkpoint、恢复以及不读取 GT 的 DDIM 推理。
两种情况下 density 均无梯度，refinement 有非零梯度。此前实际完整模型 H100 的三个
loss/backward probe 也确认了这一点，见 `loss_numerics_verification.json`；本次未启动新的 GPU 训练。

此前 500-step 日志的 NaN 记录继续保留。已完成的权重/Adam 有限性检查和空直方图数值保护
不代表训练收敛，也不改变上述几何冻结结论。

## 复查入口与输出

在仓库根目录，使用当前环境执行：

```bash
python scripts/audit_upstream_weights.py \
  --local-trained artifacts/mvwd/runs/level1_robots_overfit/checkpoints/last.ckpt
```

这一步使用 CPU、标准库和网络，不加载完整 tensor 到 RAM，也不分配 GPU。不指定
`--local-trained` 可只检查 SD 初始化与两个作者权重。可通过 `--init`、`--sd` 和 `--output`
指定另一台机器的路径。官方权重下载被限制为 HTTP range，服务器不正确支持 range 时会报错。

输出 `artifacts/mvwd/upstream_audit/weights.json`，记录初始化验签、两个作者 checkpoint 的元数据、
全量 tensor 筛查、SHA256 anchors 和可选本地训练权重比对。
同目录的 `source.json` 记录本次公开代码来源/方法比对，`summary.json` 记录核验结果和测试信息；
原始公开文件快照一并保留。`artifacts/` 仍被 Git 忽略，脚本与本文可以提交。

## 对正式实验的含义

当前版本应按公开代码的 **Top2Pano-Persp adaptation** 命名，说明其 coarse 是冻结的 BEV
VAE 特征渲染，以及额外的共享机器人资产条件。当前 2-episode / 500-step 结果属于开发检查，
不能代表正式 baseline 收敛效果；正式实验仍需固定 train/val/test、合理训练预算和验证集选模。

作者已训练 checkpoint 的 refinement 可能提供另一个初始化选择，但不会补上可训练 coarse。
采用它需要另设明确的初始化实验，核对源预训练数据与 MVWD held-out scenes 的重叠，并保持
与其他方法一致的数据使用政策。它也不能直接作为 MVWD `--resume` 文件：来源训练状态、
机器人资产 provenance 和当前 Lightning 配置不同。

若以后要求 coarse 随训练改善，需要显式另立几何训练 variant；这会改变当前按公开代码复现
的范围。本次没有进行该改动。
