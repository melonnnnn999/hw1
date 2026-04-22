## 环境要求

- 推荐使用 Python 3.10 或更高版本。
- 安装依赖见： requirements.txt


## 运行

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

最佳实验结果为：

- 隐藏层维度：`128`
- 激活函数：`relu`
- 最佳验证集准确率：`0.6136`
- 最佳轮次：`29`
- 测试集准确率：`0.6111`
- 测试集损失：`1.0892`


