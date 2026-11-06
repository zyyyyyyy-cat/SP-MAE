# WHU-MARS / UAD：SP-MAE V1.1 最终代码修改指导

实验名称：Structure-Preserving Modality-Aware Erasing（结构保留型模态感知擦除）

实施目标：只修改训练期数据增强，保留 UAD 原有模型、身份监督、DDP 训练和评测协议。

## 一、实施前的最终审查结论

我重新核对了 UAD 官方的 [`make_dataloader.py`](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/make_dataloader.py?utm_source=chatgpt.com)、[`bases.py`](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/datasets/bases.py?utm_source=chatgpt.com)、[`UAD.yml`](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/configs/UAD.yml?utm_source=chatgpt.com)，以及相关参考论文的实际方法。

结论：SP-MAE V1.1 可以按当前整体设计实施，没有发现必须推翻方案的结构性问题。

但有三个重要边界必须在实现前明确。

| 审查重点          | 最终结论                                  | 实施要求                      |
| ------------- | ------------------------------------- | ------------------------- |
| 是否与 UAD 兼容    | 兼容。新方法仅操作输入图像，不改变张量尺寸或模型参数            | 必须在数据加载阶段完成，不进入 ViT 或损失函数 |
| 是否能够保护弱模态身份信息 | 相比完全替换原像素，可以保留部分局部结构，但不能保证身份信息不受损     | 使用有界混合、单次局部变换，不能再叠加原始擦除   |
| 是否具有足够增强力度    | p=0.5、5%–20%面积、0.5–0.8混合强度可以产生明确的局部变化 | 不应额外减小增强概率，也不需要再增加其他图像增强  |

另外有一点需要修正之前的表述：

全局均值填充不一定在每张图像上都比局部均值填充产生更大变化。 如果增强区域的像素本身接近全图均值，变化依然可能很小。这不是代码错误，而是当前算子的固有性质。

本次保留全局均值，不再引入额外的动态强度控制。

## 二、论文来源与模块对应关系

这一部分必须严格区分：

SP-MAE 并不是直接复制某篇论文中名为 SP-MAE 的现成模块。它是结合四篇开源论文的研究思想，为 UAD 设计的固定数据增强算子。

因此，不能将我们的公式、全局均值填充或具体参数描述为原论文已有的方法。

| 参考论文                                       | 原文中的具体位置或模块                                                                     | 实际参考了什么                           | 是否直接移植                  |
| ------------------------------------------ | ------------------------------------------------------------------------------- | --------------------------------- | ----------------------- |
| Random Erasing Data Augmentation，AAAI 2020 | 方法部分的随机区域擦除机制                                                                   | 随机选择矩形、构造遮挡训练样本，提高对局部缺失的鲁棒性       | 部分借鉴区域采样思想              |
| COPE，CVPR 2026                             | 方法中的 Cross-Identity Consistent Occlusion（CICO） 与 Prompt Background Filling（PBF） | 遮挡不仅导致信息缺失，还可能造成特征干扰；增强设计需要重视遮挡内容 | 仅借鉴动机，没有移植 CICO/PBF     |
| PPA，CVPR 2026                              | 第3.2节 Keypoint-Preserving Regularization（KPR）                                   | 跨模态图像扰动需要考虑人体关键结构的保留              | 仅借鉴结构保护原则，不使用关键点        |
| NightReID，AAAI 2025                        | EDA 框架中的 Image Enhancement and Denoising（IED）                                   | 低质量 ReID 图像处理应关注有用细节的保留           | 仅借鉴低质量输入处理思想，不移植 IED 网络 |
| WHU-MARS / UAD，CVPR 2026                   | UAD 数据加载器、共享 ViT、ProCA/GPD                                                      | 基础训练与检索框架，直接确定改动位置                | 完整保留原框架                 |

对应论文与代码：

| 论文             | 论文链接                                                                                                                                                                                                                       | 官方代码                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Random Erasing | [AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/7000?utm_source=chatgpt.com)                                                                                                                                  | [GitHub](https://github.com/zhunzhong07/Random-Erasing?utm_source=chatgpt.com)            |
| COPE           | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_COPE_Consistent_Occlusion_and_Prompt_Enhancement_Network_for_Occluded_Person_CVPR_2026_paper.html?utm_source=chatgpt.com)                              | [GitHub](https://github.com/Cecoming/COPE?utm_source=chatgpt.com)                         |
| PPA            | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_Towards_Cross-Modal_Preservation_Consistency_and_Alignment_for_Privacy-Preserving_Visible-Infrared_Person_CVPR_2026_paper.html?utm_source=chatgpt.com) | [GitHub](https://github.com/Dige945/PPA_CVPR26?utm_source=chatgpt.com)                    |
| NightReID      | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33142?utm_source=chatgpt.com)                                                                                                                                 | [GitHub](https://github.com/msm8976/NightReID?utm_source=chatgpt.com)                     |
| WHU-MARS / UAD | [官方项目](https://github.com/msm8976/WHU-MARS?utm_source=chatgpt.com)                                                                                                                                                         | [UAD 实现](https://github.com/msm8976/WHU-MARS/tree/main/CVPR26_UAD?utm_source=chatgpt.com) |

原论文中的具体事实包括：COPE 的 CICO 对不同身份构造一致遮挡并配合特征学习，PBF 通过前景提示处理背景；PPA 的 KPR 使用姿态关键点及额外正则约束；NightReID 的 IED 是可训练的图像增强去噪模块。它们均不等同于我们的软遮挡公式。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com

+3



### SP-MAE 中各项设计的真实来源

| SP-MAE 组成               | 来源性质                         |
| ----------------------- | ---------------------------- |
| 随机矩形区域                  | 参考 Random Erasing 的区域擦除设计    |
| 重视身份结构保护                | 参考 COPE 的遮挡干扰分析、PPA 的 KPR 思想 |
| 考虑弱模态图像质量               | 参考 NightReID 的 IED 研究动机      |
| RGB 保持原配置，NIR/TIR 使用软遮挡 | 本方案针对 UAD 的适配设计              |
| 全图每通道均值混合               | 本方案设计，并非上述论文原始模块             |
| 平滑边界、0.5–0.8强度、5%–20%面积 | 本方案固定参数，尚无 WHU-MARS 性能验证     |

这样的来源表述能够避免后续论文写作时将“思想参考”误写成“复现原论文模块”。

## 三、冻结后的算法定义

SP-MAE 只作用于 NIR 和 TIR 的训练图像。

给定输入：

\\[ I\in[0,1]^{3\times256\times128} \\]

首先以概率 \\(p=0.5\\) 决定是否增强。未触发时直接返回原图。

触发时，选择一个随机矩形区域 \\(\Omega\\)，其目标面积比例和宽高比分别为：

\\[ s\sim U(0.05,0.20) \\]

\\[ r\sim\operatorname{LogUniform}(0.5,2.0) \\]

然后计算增强前整张图像的每通道均值：

\\[ \mu_c=\frac{1}{HW}\sum\_{y=1}^{H}\sum\_{x=1}^{W}I_c(x,y) \\]

构造空间软掩码 \\(M\\)，区域外为0，区域内部中心为1，边缘采用平滑过渡，并采样：

\\[ \lambda\sim U(0.5,0.8) \\]

最终：

\\[ \boxed{ I'\_c(x,y) = I_c(x,y)+\lambda M(x,y)\left[\mu_c-I_c(x,y)\right] } \\]

同一图像的三个通道共享一个 \\(M\\) 和一个 \\(\lambda\\)，但每个通道具有自己的 \\(\mu_c\\)。

这一算子有三个明确性质：输出像素仍在[0,1]，不增加模型参数，在矩形内部的平坦掩码区域中，原图局部梯度会按 \\(1-\lambda\\) 缩小而不是被随机噪声完全替换。

需要注意，软掩码过渡带可能引入新的边缘梯度，而全局均值混合可能削弱前景与背景的局部对比度。因此，这是信息保留程度更高的遮挡增强，不是保证保留身份结构的算法。

## 四、完整代码修改范围

推荐只修改训练数据处理相关的三个文件，并增加一组配置项。

| 文件                                       | 操作    | 具体职责                                             |
| ---------------------------------------- | ----- | ------------------------------------------------ |
| `CVPR26_UAD/datasets/sp_mae.py`          | 新增    | 实现 SP-MAE 的区域采样、均值计算、软掩码和像素混合                    |
| `CVPR26_UAD/datasets/make_dataloader.py` | 修改    | 为 RGB、IR、Thermal 构造对应的训练 Transform               |
| `CVPR26_UAD/datasets/bases.py`           | 少量修改  | 在训练集 `ImageDataset.__getitem__` 中按模态选择 Transform |
| 当前使用的 UAD 实验配置文件                         | 新增配置项 | 控制增强开关及固定参数                                      |

不需要修改 `train.py`、`processor`、ViT Backbone、ProCA、GPD、WRT 或评测代码。

这里的路径对应官方仓库结构。如果本地已经进行了 DDP 适配，应以本地正在使用的 UAD 代码为基础修改，而不是用官方旧文件覆盖现有实现。

## 五、新增 `datasets/sp_mae.py`

这个文件只负责一个功能：输入单张已转换为 Tensor 的图像，返回经过 SP-MAE 处理的同尺寸图像。

它不需要知道 PID、Camera ID、Ground/Aerial 平台，不参与模型训练，也不维护任何可学习参数。

### 5.1 模块输入与输出

| 属性   | 具体要求                              |
| ---- | --------------------------------- |
| 输入   | `torch.Tensor`，形状 `[3,H,W]`       |
| 输入数值 | `[0,1]`，尚未 Normalize              |
| 输出   | `torch.Tensor`，形状、dtype、device 不变 |
| 增强概率 | 0.5                               |
| 新增参数 | 0                                 |
| 增强次数 | 最多1次                              |
| 调用位置 | 训练期 `ToTensor` 后、`Normalize` 前    |

不支持在标准化后的数据上执行该算子。

例如，UAD 的 `Normalize(mean=0.5,std=0.5)` 会把输入转换到约[-1,1]，而 SP-MAE 的数学边界及混合定义使用的是[0,1]。将顺序颠倒，会使实现偏离设计。

### 5.2 模块内部执行顺序

输入 Tensor：`[3,256,128]`，值域[0,1]

概率判断 p=0.5

未触发直接返回原图

采样一个合法矩形区域 Ω

面积5%–20%，对数均匀宽高比0.5–2.0

计算增强前的每通道全图均值 μ

创建二维软掩码 M

矩形中心为1，边缘逐渐降为0

采样混合系数 λ∈[0.5,0.8]

有界像素混合 → 输出增强图像

需要特别明确以下实现细节。

A. 区域采样

矩形尺寸根据目标面积和宽高比计算，之后随机选择左上角位置。区域必须位于图像内部，不能因越界而扩大或裁掉增强区域。

建议最多尝试10次选择合法矩形。如果未成功，则返回原图。不要为了满足触发概率而强行生成一个不符合面积约束的矩形。

区域面积5%–20%指矩形包围区域的面积比例，由于存在平滑边界，最终受到最大混合强度影响的有效面积会略小。

B. 全局均值

均值来自当前已经完成 Resize、Flip、Padding 和 Crop 的图像，且在软遮挡前计算。

每通道分别计算一个值，形状应能够广播至 `[3,H,W]`。

不要误用整个训练集的统计均值，也不要直接把 `PIXEL_MEAN=[0.5,0.5,0.5]` 当作填充值。

C. 软掩码

推荐将平滑边界定义为：

- 矩形四边内侧各设置宽度约为对应边长10%的过渡带；
- 从矩形边界向中心进行0到1的线性过渡；
- 中心剩余区域保持1；
- 水平、垂直方向分别构建一维权重，再相乘得到二维掩码；
- 矩形外严格为0。

这样可以保证实现具有确定含义，而不是不同开发人员各自选择高斯核、模糊半径或其他平滑方法。

D. 混合系数

整张图像只采样一个 \\(\lambda\\)，同时作用于该图的三个通道。

不要逐通道独立采样，不要为每个像素独立采样，也不要在同一图像中重复应用多次。

E. 数值与缓冲区

保持原图 dtype/device，不要产生不必要的 CPU/GPU 转移。

推荐使用安全的张量操作，避免因原地修改影响后续均值计算或复用同一图像缓冲区。

正常情况下，凸组合的输出应自然位于[0,1]；不应该依赖最后的大量 `clamp` 掩盖错误运算。

## 六、修改 `datasets/make_dataloader.py`

这是实际集成的核心文件。

### 6.1 保留原始 UAD 几何增强

原始训练流水线为：

`Resize → RandomHorizontalFlip → Padding → RandomCrop → ToTensor → Normalize → RandomErasing`

新方案不改变前面五项操作，只根据模态决定后面的增强方式。

### 6.2 建立三种训练 Transform

| 模态        | 训练 Transform                           |
| --------- | -------------------------------------- |
| `RGB`     | 原始 UAD 完整 Transform                    |
| `IR`      | 原始几何操作 → ToTensor → SP-MAE → Normalize |
| `Thermal` | 原始几何操作 → ToTensor → SP-MAE → Normalize |

特别注意：

对于 `IR` 和 `Thermal`，原来的 `RandomErasing(mode="pixel")` 必须被替换，不能保留在 SP-MAE 后面。

对于 RGB，则应完整保留原始 Random Erasing：

- `probability=cfg.INPUT.RE_PROB`；
- `mode="pixel"`；
- `max_count=1`；
- `device="cpu"`；
- 其余默认采样参数不变。

不建议同时为 RGB 修改任何增强参数，否则就不再是仅针对弱模态擦除方式的对照实验。

### 6.3 保证三个模态的基础随机增强逻辑不变

UAD 的 `ImageDataset` 会按 PKM 采样得到的各模态索引读取对应图像，并分别调用 Transform。

这里要保留原来的行为：同一身份的 RGB、IR、Thermal 图像仍各自独立进行随机翻转、裁剪和增强采样。

不要因为构造了三个 Transform，就强行共享同一身份三张图像的裁剪参数或软遮挡位置。

这会额外改变原始训练数据分布，不属于本次方案。

### 6.4 不修改验证 Transform

UAD 的 `val_transforms` 仍然保持：

`Resize → ToTensor → Normalize`

不添加 SP-MAE，不添加 Random Erasing，也不改变测试插值方式。

这一点适用于验证集、Query、Gallery，以及原有 `train_loader_normal`。

尤其不要误改 `train_loader_normal`。 它在 UAD 中使用正常图像 Transform，与主训练增强数据流不是同一个入口。本次需要保持其原始行为，以免将数据增强修改扩大到原型或正常特征统计相关流程。

## 七、修改 `datasets/bases.py`

官方 `ImageDataset` 当前接收统一的 `transform`，并对所有模态调用它。

本次需要将其扩展为支持根据真实模态名称选择训练 Transform。

推荐的上层实现方式是：

将 `transform` 允许设置为模态到增强流水线的映射；训练集在读取每张图像时，根据当前循环取得的 `modality` 选择对应的 Transform。

模态名称严格使用官方配置：

| 官方标识      | 数据含义    | 使用的增强  |
| --------- | ------- | ------ |
| `RGB`     | 可见光     | 原始 UAD |
| `IR`      | 近红外 NIR | SP-MAE |
| `Thermal` | 热红外 TIR | SP-MAE |

另外要保持原来的单个 Transform 使用方式，以兼容其他沿用此 Dataset 类的调用路径。

不要修改 `read_image()` 的图像解码和 `.convert("RGB")` 逻辑。

虽然 NIR 可能来自单通道灰度图，但 UAD 原有网络输入是三通道。本次应保留原始读图协议，而不是顺带将 NIR 改为单通道输入。

需要明确的是，`.convert("RGB")` 不意味着原始热红外具有三个独立光谱通道。SP-MAE 只是处理 UAD 当前读取到的三通道张量，不重新定义数据的物理含义。

## 八、增加实验配置

建议在当前实际使用的 UAD 配置中增加独立的 SP-MAE 配置区，参数名称可以由开发者统一确定，但含义和值必须固定。

| 配置                           | 冻结值                    |
| ---------------------------- | ---------------------- |
| `SP_MAE.ENABLED`             | `True`                 |
| `SP_MAE.MODALITIES`          | `["IR","Thermal"]`     |
| `SP_MAE.PROB`                | `0.5`                  |
| `SP_MAE.MIN_AREA`            | `0.05`                 |
| `SP_MAE.MAX_AREA`            | `0.20`                 |
| `SP_MAE.MIN_ASPECT`          | `0.5`                  |
| `SP_MAE.MAX_ASPECT`          | `2.0`                  |
| `SP_MAE.ASPECT_DISTRIBUTION` | `"log_uniform"`        |
| `SP_MAE.MIN_STRENGTH`        | `0.5`                  |
| `SP_MAE.MAX_STRENGTH`        | `0.8`                  |
| `SP_MAE.MAX_REGIONS`         | `1`                    |
| `SP_MAE.EDGE_FRACTION`       | `0.10`                 |
| `SP_MAE.FILL_TARGET`         | `"image_channel_mean"` |

这里的配置名称是建议的接口命名，不代表原仓库已有这些字段。应沿用本地项目现有的配置组织方式。

建议支持通过一个开关启用/关闭 SP-MAE，以便复用同一份代码运行原始 UAD。除此之外，不需要额外提供多档实验参数，也不要预设自动调参功能。

## 九、对 UAD 模型和训练过程的影响

这部分的原则非常明确：允许输入图像发生变化，但不允许在实现时顺便修改学习框架。

| 模块       | 是否修改 | 注意事项                                         |
| -------- | ---- | -------------------------------------------- |
| ViT-B/16 | 否    | 输入仍为 `[3,256,128]`，Patch Size、Token 数、位置编码不变 |
| BNNeck   | 否    | 原始训练与测试使用方式不变                                |
| CE / WRT | 否    | 不增加新的监督项                                     |
| ProCA    | 否    | 原有身份中心及更新逻辑不变                                |
| GPD      | 否    | 原有全局原型及相关计算不变                                |
| PKM      | 否    | 身份采样、模态组合不变                                  |
| 优化器与学习率  | 否    | 与已复现的 UAD 基线一致                               |
| 推理特征     | 否    | 继续使用 `NECK_FEAT=before`                      |
| 测试距离     | 否    | L2 归一化后欧氏距离，原有同相机过滤                          |

需要注意：虽然不修改 ProCA/GPD，但增强后的图像会改变进入这些模块的训练特征，因而可能改变身份中心和原型的学习结果。

这属于本次实验希望观察的自然结果，不需要额外调整它们的权重来适应 SP-MAE。

### 对 DDP 的额外注意事项

在原 UAD 已有的 DDP 代码基础上，只需注意三点。

第一，SP-MAE 是无参数的数据增强操作，通常在 DataLoader Worker 中执行，不需要加入 DDP 模型包装，也不需要额外梯度同步。

第二，每张训练图像应独立采样增强参数。不同 GPU、不同 Worker 不需要共享同一矩形或混合系数。沿用项目现有的随机种子管理方式即可，避免意外让各 Worker 获得完全相同的随机序列。

第三，不改变 `PKMSampler_DDP`、BatchSampler、有效 Batch、训练轮次或梯度更新次数。增强操作不应该影响身份标签、模态顺序或 Batch 拼接。

不需要为这个简单增强模块重新设计 DDP。

## 十、实施前的必要正确性检查

不需要开展新的大型诊断实验。只做以下一次性核查即可。

| 核查项   | 必须满足的结果                                           |
| ----- | ------------------------------------------------- |
| 模态分流  | RGB 使用原增强；IR、Thermal 使用 SP-MAE                    |
| 输入输出  | 均为 `[3,256,128]`，dtype/device 保持一致                |
| 像素数值  | SP-MAE 输出处于[0,1]，无 NaN/Inf                        |
| 灰度一致性 | 对三通道完全相同的 NIR 输入，增强后仍然相同                          |
| 增强有效性 | 被触发时，除特殊均值相近区域外，存在实际局部变化                          |
| 增强范围  | 矩形区域外不发生变化                                        |
| 训练路径  | IR/Thermal 不再执行原始硬随机擦除                            |
| 非训练路径 | `train_loader_normal`、验证、Query/Gallery 不执行 SP-MAE |
| 模型兼容  | 原始 UAD 前向输出与损失计算正常                                |
| DDP   | 各 GPU 正常读取数据和执行训练，无额外参数同步需求                       |

建议分别保存少量 RGB、NIR、TIR 原图及增强图用于肉眼核查。

这里重点看的是是否出现错误现象，例如整个图像被改变、伪彩色异常、矩形外像素变化或前景大面积变成统一常数，而不是再进行复杂的图像质量统计。

## 十一、正式实验设置

本次只运行一个固定候选。

| 实验          | 内容                        | 比较目的                  |
| ----------- | ------------------------- | --------------------- |
| B0          | 已复现的原始 UAD                | 基准                    |
| SP-MAE V1.1 | RGB 原样；NIR/TIR 替换为结构保留软遮挡 | 判断改进弱模态遮挡方式能否提升统一身份检索 |

建议先使用 WHU-MARS-1000，严格复用 B0 的训练与评测设置。



最重要的是，不要在看到结果后继续调整面积、概率、强度或填充值来进行无终点的参数搜索。这次已经冻结了完整设计。