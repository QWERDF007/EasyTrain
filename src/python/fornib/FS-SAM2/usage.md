# FS-SAM2 训练指南

## 数据准备


### custom 数据集结构

```
your_dataset/
├── cat/
│   ├── images/
│   │   ├── cat1.jpg
│   │   └── cat2.jpg
│   ├── masks/
│   │   ├── cat1.png      # 二值掩码，0=背景，255=目标
│   │   └── cat2.png
│   ├── support.txt        # 支撑集图像名（不含扩展名），每行一个
│   └── query.txt          # 查询集图像名（不含扩展名），每行一个
├── dog/
│   ├── images/
│   ├── masks/
│   ├── support.txt
│   └── query.txt
└── ...
```

- 图像格式：`.jpg` / `.jpeg` / `.png`
- 掩码格式：`.png`，与图像同名，二值（阈值 128）
- support.txt / query.txt 每行一个图像名（不含扩展名），不可重叠
- 前 80% 类（按字母序）作为训练集，后 20% 作为验证集

### 数据量要求

| 维度 | 最低 | 建议 |
|------|------|------|
| 类别数 | ≥2 | ≥5 |
| 每类 support 图像 | ≥kshot | 3-5 张 |
| 每类 query 图像 | ≥1 | 3-5 张 |


### LabelMe 转 custom 格式

```bash
python labelme2custom.py \
    --image_dir /path/to/images/ \
    --json_dir /path/to/labelme_jsons/ \
    --output_dir /path/to/custom_dataset/ \
    --support_ratio 0.5
```

- `--image_dir`：图像文件夹
- `--json_dir`：LabelMe `.json` 标注文件夹
- `--output_dir`：输出的 custom 数据集目录
- `--support_ratio`：每类图像中用作 support 的比例，默认 0.5


## 环境准备

```bash
# conda 环境
conda activate py312

# 进入 FS-SAM2 目录
cd F:\Projects\EasyTrain\src\python\fornib\FS-SAM2

# 下载 SAM 2.1 权重
wget -P checkpoint https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

## 训练

```bash
python -m torch.distributed.run --nproc_per_node=1 train.py \
    --datapath /path/to/your_dataset/ \
    --benchmark custom \
    --kshot 1 \
    --epochs 50 \
    --lr 1e-4 \
    --bsz 2 \
    --nworker 0 \
    --exp_id 0000 \
    --fold 0 \
    --sam2_checkpoint ./checkpoint/sam2.1_hiera_base_plus.pt \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--datapath` | 数据集根目录 |
| `--benchmark custom` | 使用 custom 数据集 |
| `--kshot` | 每类 support 样本数 (1~5) |
| `--epochs` | 训练轮数 |
| `--lr` | 学习率 |
| `--bsz` | 单卡 batch size |
| `--nworker` | CPU 线程数 (Windows 必须设为 0) |
| `--exp_id` | 实验 ID |
| `--fold` | custom 数据集忽略此参数，填 0 即可 |
| `--sam2_checkpoint` | SAM2 模型权重路径 |
| `--sam2_cfg` | SAM2 模型配置文件路径 |

SAM2 模型可选：

| 模型 | checkpoint | cfg |
|------|-----------|-----|
| Hiera Base+ (默认) | `.../sam2.1_hiera_base_plus.pt` | `.../sam2.1_hiera_b+.yaml` |
| Hiera Large | `.../sam2.1_hiera_large.pt` | `.../sam2.1_hiera_l.yaml` |
| Hiera Tiny | `.../sam2.1_hiera_tiny.pt` | `.../sam2.1_hiera_t.yaml` |

### 输出

- 日志：`logs/custom/0000/fold0.log/`
- 最佳模型：`logs/custom/0000/fold0.log/best_model.pt`
- TensorBoard：`logs/custom/0000/fold0.log/tbd/runs/`

### 训练日志解读

```
*** Training [@Epoch 00] Avg L: 0.51274  mIoU: 41.87 (41.87)  FB-IoU: 70.50   mF1: 59.02 (59.02)  ***
```

| 字段         | 含义                                                       |
| ------------ | ---------------------------------------------------------- |
| **Epoch**    | 第几轮训练                                                 |
| **Batch**    | 当前 batch / 总 batch 数                                   |
| **L** (Loss) | 当前 batch 的 BCE + Dice 损失                              |
| **Avg L**    | 当前 epoch 累计平均损失                                    |
| **mIoU**     | mean Intersection over Union，平均交并比。括号内是每类逐项 |
| **FB-IoU**   | Foreground-Background IoU，前后景全局交并比                |
| **mF1**      | mean F1 Score，精确率+召回率的调和平均。括号内是每类逐项   |
| **dt**       | 耗时（秒）                                                 |

- Training 和 Validation 会分别打印
- 验证 mIoU 远低于训练 mIoU 说明过拟合，通常是因为数据量太少
- FB-IoU 反映模型区分前后景的能力，mIoU 反映具体轮廓精度



## 测试

```bash
python test.py \
    --datapath /path/to/your_dataset/ \
    --benchmark custom \
    --kshot 1 \
    --logpath logs/custom/0000/fold0/ \
    --nworker 0 \
    --sam2_checkpoint ./checkpoint/sam2.1_hiera_base_plus.pt \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml
```



## 预测

```bash
python predict.py \
    --support_dir /path/to/support/ \
    --query_dir /path/to/queries/ \
    --output_dir /path/to/output/ \
    --checkpoint logs/custom/0000/fold0.log/best_model.pt \
    --sam2_checkpoint ./checkpoint/sam2.1_hiera_base_plus.pt \
    --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
    --kshot 1
```

support 目录结构：

```
support_dir/
├── images/
│   ├── example1.jpg
│   └── example2.jpg
└── masks/
    ├── example1.png    # 二值掩码
    └── example2.png
```

### 为何需要 support

FS-SAM2 是 **Few-Shot Semantic Segmentation**，不是固定类别的语义分割。

### 训练学了什么

LoRA 微调学的是**"如何从示例学习"**这个通用能力，不是某个具体类别的特征。

- 训练时 80% 的类用于训练，20% 的类模型从未见过
- 验证/测试就是在测模型对全新类的泛化
- LoRA 学的是"匹配 support 特征到 query 图"的机制

### 推理时为什么需要 support

模型不知道你要分割什么。同一份权重，给不同的 support 就分割不同的东西：

```
support: [猫图 + 猫mask]     →  query: [新图]  →  输出猫的mask
support: [缺陷图 + 缺陷mask]   →  query: [新图]  →  输出缺陷的mask
```

### 预测时 support 数量

没有硬性限制，predict.py 会使用 `support_dir` 下所有图像。

- 至少 1 张，必须有对应 mask
- **与训练时 `--kshot` 一致效果最好**（如训练 kshot=1，预测也用 1 张 support）
- 用多张 support 也能跑，模型支持累积多张 support memory，但效果不一定提升（训练时没学过融合多个 support）


### 如果不需要 support

如果分割目标是**固定的**已知类别，应该用传统语义分割（closed-set），如 DeepLab、SegFormer 等。FS-SAM2 不适合这个场景。



