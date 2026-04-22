# HW1：使用 NumPy 实现 EuroSAT 三层 MLP 分类器

这个文件夹已经按照适合上传到 GitHub 仓库的形式整理好了。代码按模块拆分，运行命令可复现，同时把数据集和模型权重这类较大的文件排除在版本管理之外。

## 仓库内容

- 使用 NumPy 从零实现三层 MLP，用于 EuroSAT 遥感图像地表覆盖分类。
- 按数据处理、模型定义、训练、评估、超参数搜索、可视化拆分好的代码模块。
- 包含环境依赖说明和运行命令的 README。
- 适合公开放在 GitHub 上的轻量实验结果文件。

## 推荐目录结构

```text
hw1_github_repo/
|- hw1_mlp/
|  |- data.py
|  |- model.py
|  |- metrics.py
|  |- runner.py
|  |- visualize.py
|  `- cli.py
|- main.py
|- requirements.txt
|- .gitignore
`- results/
   |- final_report/
   `- search_report/
```

## 环境要求

- 推荐使用 Python 3.10 或更高版本。
- 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集说明

数据集不建议上传到 GitHub。请先在本地下载并解压 EuroSAT RGB 数据集，然后将其放在仓库根目录下：

```text
EuroSAT_RGB/
```

如果数据集不在仓库根目录，也可以通过 `--data-dir` 参数手动指定路径。

## 运行方法

### 1. 超参数搜索

```bash
python main.py search \
  --data-dir EuroSAT_RGB \
  --output-dir runs/search_report \
  --max-per-class 300 \
  --epochs 3 \
  --hidden-dims 64,128 \
  --lrs 0.005,0.001 \
  --weight-decays 0,0.0001 \
  --activations relu,tanh
```

### 2. 训练最终模型

```bash
python main.py train \
  --data-dir EuroSAT_RGB \
  --output-dir runs/final_relu128_e30 \
  --epochs 30 \
  --hidden-dim 128 \
  --activation relu \
  --lr 0.005 \
  --lr-decay 0.95 \
  --weight-decay 0.0001 \
  --batch-size 128 \
  --seed 42
```

### 3. 在测试集上评估

```bash
python main.py test \
  --data-dir EuroSAT_RGB \
  --checkpoint runs/final_relu128_e30/best_model.npz \
  --split test
```

### 4. 绘制训练曲线

```bash
python main.py plot \
  --history runs/final_relu128_e30/history.json \
  --output runs/final_relu128_e30/training_curves.png
```

### 5. 可视化第一层权重

```bash
python main.py weights \
  --checkpoint runs/final_relu128_e30/best_model.npz \
  --output runs/final_relu128_e30/first_layer_weights.png \
  --num 12
```

### 6. 生成错例分析结果

```bash
python main.py errors \
  --data-dir EuroSAT_RGB \
  --checkpoint runs/final_relu128_e30/best_model.npz \
  --output-dir runs/final_relu128_e30/error_examples \
  --num 8
```

## 已包含的报告结果文件

`results/` 文件夹中已经放入从最终实验结果里整理出来的轻量文件：

- `results/final_report/training_curves.png`
- `results/final_report/first_layer_weights.png`
- `results/final_report/test_result.txt`
- `results/final_report/test_confusion_matrix.csv`
- `results/final_report/test_metrics.json`
- `results/final_report/history.json`
- `results/final_report/result_summary.txt`
- `results/final_report/error_examples/wrong_examples.png`
- `results/final_report/error_examples/wrong_examples.txt`
- `results/search_report/search_results.csv`
- `results/comparison_summary.csv`

这个可直接上传的仓库版本中，当前记录的最佳实验结果为：

- 隐藏层维度：`128`
- 激活函数：`relu`
- 最佳验证集准确率：`0.6136`
- 最佳轮次：`29`
- 测试集准确率：`0.6111`
- 测试集损失：`1.0892`

## 不建议上传到 GitHub 的内容

- `EuroSAT_RGB/` 数据集
- `best_model.npz` 这类较大的模型权重文件
- 最终模型权重建议上传到 Google Drive 或其他网盘，再把下载链接补到实验报告和 README 中

## 提交前需要补充的链接

- GitHub Repo 链接：替换成你自己的 Public GitHub 仓库地址
- 模型权重下载链接：替换成你上传到 Google Drive 或其他网盘后的分享链接
