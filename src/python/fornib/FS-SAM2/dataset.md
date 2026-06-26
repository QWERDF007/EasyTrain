# FS-SAM2 数据集 Train/Val 划分

## Custom

- 类名排序后按 **80/20** 切分：前 80% 类 = train，后 20% 类 = val
- 每类内部由 `support.txt` / `query.txt` 定义 support（训练时用作 memory）和 query（待分割图像）
- 单类时 val 回退使用全部类
- train/val 类别不重叠（单类回退除外）

## COCO-20i

- 4-fold 交叉验证
- 80 类 / 4 = 20，每个 fold 取 **20 类** 做 val，其余 **60 类** 做 train
- 预生成 split 文件：`data/splits/lists/coco/fss_list/{train,val}/`
- train/val 类别完全不重叠

## PASCAL-5i

- 4-fold 交叉验证
- 20 类 / 4 = 5，每个 fold 取 **5 类** 做 val，其余 **15 类** 做 train
- 预生成 split 文件：`data/splits/lists/pascal/fss_list/{train,val}/`
- train/val 类别完全不重叠
- fold 10-13 为 COCO→PASCAL 跨域测试

## FSS-1000

- 固定三类划分：**0–519** = train，**520–759** = val，**760–999** = test
- 预定义文件：`data/splits/lists/fss/{trn,val,test}.txt`
- 每类固定 10 张图，其中 1 张 query + (1~9) support

## MVTec-Unseen

- 5-fold 交叉验证
- 5 大类 × 5 种缺陷 = 25 类，每个 fold 取不相交子集做 val
- train = 正常样本打 mask，val/test = 未见缺陷做 novel class
