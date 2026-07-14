# Household Power Consumption Forecasting

本项目用于家庭用电负荷的多步时间序列预测，支持以下三类模型：

- **LSTM**
- **Transformer**
- **ADLG-HFormer**

训练脚本支持两种预测任务、多随机种子重复实验、模型检查点保存，以及缩放空间和原始数值空间下的 MSE/MAE 评估。

## 1. 主要功能

- 支持 `90 → 90` 和 `90 → 365` 两种预测任务。
- 统一训练 LSTM、Transformer 和 ADLG-HFormer。
- 使用 MSE 作为训练损失。
- 训练过程中执行梯度裁剪，降低梯度爆炸风险。
- 自动保存每轮 checkpoint、最新模型和最优模型。
- 支持多随机种子重复实验。
- 自动统计 MSE、MAE 的均值和标准差。
- 若提供 `target_scaler.pkl`，可将预测值恢复到原始数值尺度后计算指标。
- 保存实验配置、逐 epoch 日志和最终汇总结果。

## 2. 项目结构

建议将项目组织为：

```text
household/
├── dataset/
│   ├── __init__.py
│   └── dataset.py
├── modeling/
│   ├── __init__.py
│   ├── LSTM.py
│   ├── transformer.py
│   └── ADLGHFormer.py
├── data/
│   ├── train/
│   │   └── target_scaler.pkl
│   └── eval/
├── outputs/
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

其中：

- `dataset/dataset.py` 需要提供 `build_dataloader`。
- `modeling/LSTM.py` 需要提供 `LSTMForecaster`。
- `modeling/transformer.py` 需要提供 `TransformerForecaster`。
- `modeling/ADLGHFormer.py` 需要提供 `ADLG_HFormer`。
- `train.py` 表示当前训练脚本；若脚本文件名不同，请替换后续命令中的文件名。

## 3. 环境要求

推荐环境：

- Python 3.9 或更高版本
- PyTorch
- NumPy
- pandas
- tqdm
- joblib
- scikit-learn

安装基础依赖：

```bash
pip install torch numpy pandas tqdm joblib scikit-learn
```

也可以创建 `requirements.txt`：

```text
torch
numpy
pandas
tqdm
joblib
scikit-learn
```

然后执行：

```bash
pip install -r requirements.txt
```

## 4. 数据准备

原始家庭用电数据文件可放置在本地数据目录中，例如：

```text
data/raw/household_power_consumption.txt
```

该数据文件体积超过 GitHub 普通仓库的单文件限制，建议在 `.gitignore` 中加入：

```gitignore
household_power_consumption.txt
data/raw/
outputs/
*.pt
__pycache__/
```

请不要直接将原始大文件提交到普通 Git 仓库。

训练脚本通过以下接口加载数据：

```python
from dataset.dataset import build_dataloader
```

调用形式为：

```python
build_dataloader(
    root_dir=...,
    split="train" or "eval",
    task="90to90" or "90to365",
    batch_size=...,
    shuffle=...,
    num_workers=...
)
```

因此，实际的数据预处理方式、输入文件名称和张量格式由 `dataset/dataset.py` 决定。

脚本期望：

```text
train_dataset.X.shape[-1] -> 输入特征维度
train_dataset.y.shape[-1] -> 预测长度
```

模型输入和标签通常应满足：

```text
x: [batch_size, input_length, feature_dim]
y: [batch_size, prediction_length]
```

### Target Scaler

若需要在原始用电数值尺度下计算 MSE 和 MAE，应提供：

```text
data/train/target_scaler.pkl
```

默认查找路径为：

```text
<root_dir>/data/train/target_scaler.pkl
```

也可以通过参数显式指定：

```bash
--target_scaler_path /path/to/target_scaler.pkl
```

如果找不到 scaler，程序仍可运行，但只会输出缩放空间下的指标。

## 5. 修改项目根目录

当前脚本中包含固定路径：

```python
ROOT_DIR = "/mnt/sdc/tnchen/matchine_learning"
```

运行前应将其修改为当前项目的实际路径，例如：

```python
ROOT_DIR = "/mnt/sdc/tnchen/household"
```

同时，推荐在运行命令中显式传入：

```bash
--root_dir /mnt/sdc/tnchen/household
```

## 6. 快速开始

### 6.1 运行 ADLG-HFormer

执行 `90 → 365` 预测任务：

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to365 \
  --model_type adlg_hformer \
  --device cuda:0
```

### 6.2 运行 LSTM

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to365 \
  --model_type lstm \
  --hidden_dim 128 \
  --num_layers 3 \
  --device cuda:0
```

### 6.3 运行 Transformer

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to365 \
  --model_type transformer \
  --d_model 128 \
  --nhead 4 \
  --num_layers 3 \
  --dim_feedforward 256 \
  --pooling last \
  --device cuda:0
```

### 6.4 运行 `90 → 90` 任务

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to90 \
  --model_type adlg_hformer \
  --device cuda:0
```

## 7. 多随机种子实验

默认重复运行 5 次，随机种子为：

```text
2026, 2027, 2028, 2029, 2030
```

运行命令：

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to365 \
  --model_type adlg_hformer \
  --repeat 5 \
  --seed_list 2026,2027,2028,2029,2030 \
  --device cuda:0
```

程序会输出：

- MSE mean
- MSE std
- MAE mean
- MAE std

若成功加载 `target_scaler.pkl`，汇总结果优先使用原始尺度指标；否则使用缩放空间指标。

### 只运行一个随机种子

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to365 \
  --model_type adlg_hformer \
  --repeat 1 \
  --seed 2026 \
  --device cuda:0
```

## 8. 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--root_dir` | `/mnt/sdc/tnchen/matchine_learning` | 项目根目录 |
| `--task` | `90to365` | 预测任务，可选 `90to90`、`90to365` |
| `--model_type` | `adlg_hformer` | 模型，可选 `lstm`、`transformer`、`adlg_hformer` |
| `--epochs` | `20` | 训练轮数 |
| `--batch_size` | `32` | 批大小 |
| `--lr` | `5e-4` | 学习率 |
| `--weight_decay` | `1e-5` | 权重衰减 |
| `--hidden_dim` | `128` | LSTM 隐藏层维度 |
| `--d_model` | `128` | Transformer/ADLG-HFormer 表征维度 |
| `--nhead` | `4` | 多头注意力头数 |
| `--dim_feedforward` | `256` | 前馈网络隐藏维度 |
| `--num_layers` | `3` | 模型层数 |
| `--dropout` | `0.1` | Dropout 比例 |
| `--pooling` | `last` | Transformer 池化方式，可选 `last`、`mean` |
| `--num_workers` | `4` | DataLoader 工作进程数 |
| `--repeat` | `5` | 重复实验次数 |
| `--seed_list` | `2026,...,2030` | 多轮实验随机种子 |
| `--seed` | `2026` | 单轮实验随机种子 |
| `--target_scaler_path` | `None` | 目标变量 scaler 路径 |
| `--device` | `cuda:1` | 运行设备 |
| `--output_dir` | `None` | 自定义输出目录 |

## 9. 输出目录

默认输出路径为：

```text
<root_dir>/outputs/<model_type>/<task>/<timestamp>/
```

示例：

```text
outputs/
└── adlg_hformer/
    └── 90to365/
        └── 20260714_210000/
            ├── experiment_config.json
            ├── final_eval_summary.csv
            ├── final_eval_summary_report.csv
            ├── seed_2026/
            │   ├── config.json
            │   ├── train_log.csv
            │   ├── latest.pt
            │   ├── best.pt
            │   └── checkpoint_epoch_001.pt
            ├── seed_2027/
            └── ...
```

### 单个随机种子目录

每个 `seed_xxxx` 目录包含：

- `config.json`：当前随机种子的完整配置。
- `train_log.csv`：逐 epoch 训练与验证指标。
- `latest.pt`：最新 epoch 的 checkpoint。
- `best.pt`：验证集 scaled MSE 最低的 checkpoint。
- `checkpoint_epoch_XXX.pt`：每个 epoch 的 checkpoint。

### 实验级汇总文件

- `experiment_config.json`：本次多轮实验的参数。
- `final_eval_summary.csv`：每个随机种子的最优结果。
- `final_eval_summary_report.csv`：MSE/MAE 的 mean 和 std。

## 10. 指标说明

训练和验证阶段均计算：

- **MSE**：Mean Squared Error
- **MAE**：Mean Absolute Error

脚本会记录两类指标：

### Scaled Metrics

在归一化或标准化后的数值空间中计算：

```text
train_mse_scaled
train_mae_scaled
eval_mse_scaled
eval_mae_scaled
```

### Original Metrics

通过 `target_scaler.inverse_transform` 恢复到原始数值空间后计算：

```text
eval_mse_original
eval_mae_original
```

最优模型由验证集上的 `eval_mse_scaled` 决定。

## 11. Checkpoint 内容

每个 checkpoint 包含：

```text
epoch
seed
model_state_dict
optimizer_state_dict
train_mse_scaled
train_mae_scaled
eval_mse_scaled
eval_mae_scaled
eval_mse_original
eval_mae_original
args
```

加载示例：

```python
import torch

checkpoint = torch.load(
    "outputs/adlg_hformer/90to365/<timestamp>/seed_2026/best.pt",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model_state_dict"])

print("Best epoch:", checkpoint["epoch"])
print("Original MSE:", checkpoint["eval_mse_original"])
print("Original MAE:", checkpoint["eval_mae_original"])
```

## 12. 实验复现性

脚本会同时设置：

```text
Python random seed
NumPy random seed
PyTorch CPU seed
PyTorch CUDA seed
```

并配置：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

这有助于提高实验复现性。不同硬件、CUDA、cuDNN 或 PyTorch 版本之间仍可能存在轻微差异。

## 13. 注意事项

### CUDA 设备编号

脚本默认设备为：

```text
cuda:1
```

只有一张 GPU 时，应显式指定：

```bash
--device cuda:0
```

否则可能出现：

```text
CUDA error: invalid device ordinal
```

可用以下命令查看 GPU：

```bash
nvidia-smi
```

### Checkpoint 占用空间

当前脚本会在每个 epoch 保存一个 checkpoint。训练轮数和重复次数较大时，可能占用较多磁盘空间。

只需要最优模型时，可以修改 `save_checkpoint`，不再保存：

```text
checkpoint_epoch_XXX.pt
```

### 原始指标异常

若原始尺度下的 MSE/MAE 明显异常，请检查：

- `target_scaler.pkl` 是否与当前标签使用同一训练集拟合。
- scaler 是否仅针对目标变量拟合。
- 标签维度是否与 inverse transform 逻辑一致。
- 训练集、验证集是否使用相同的数据缩放规则。

## 14. 推荐实验命令

### ADLG-HFormer：90 → 365，5 次实验

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to365 \
  --model_type adlg_hformer \
  --epochs 20 \
  --batch_size 32 \
  --lr 5e-4 \
  --weight_decay 1e-5 \
  --d_model 128 \
  --nhead 4 \
  --num_layers 3 \
  --dim_feedforward 256 \
  --dropout 0.1 \
  --repeat 5 \
  --seed_list 2026,2027,2028,2029,2030 \
  --device cuda:0
```

### LSTM：90 → 90，5 次实验

```bash
python train.py \
  --root_dir /mnt/sdc/tnchen/household \
  --task 90to90 \
  --model_type lstm \
  --epochs 20 \
  --batch_size 32 \
  --lr 5e-4 \
  --hidden_dim 128 \
  --num_layers 3 \
  --dropout 0.1 \
  --repeat 5 \
  --seed_list 2026,2027,2028,2029,2030 \
  --device cuda:0
```

## 15. License

请根据项目实际用途添加许可证，例如 MIT License。
