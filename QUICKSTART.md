# BCSS Semantic Segmentation - 完整重構指南

## 📌 項目概述

已成功將 Lab4 Jupyter notebook 重構為**生產級 Python 項目**，包含：
- ✅ 2 個獨立的、完整功能的入口點
- ✅ 模塊化架構（易於維護和測試）
- ✅ 完整的文檔和使用指南
- ✅ 自動化檢查點保存
- ✅ TensorBoard 實時監控

---

## 🎯 兩個入口點

### 1. **訓練入口點** (`train_main.py`)
完整的訓練和驗證流程，包括 Kaggle 提交。

```bash
# 完整訓練（含 Kaggle 上傳）
python train_main.py

# 訓練但不上傳
python train_main.py --no-submit
```

**功能**:
- 🔄 100 epochs 訓練（自動早期停止）
- ✓ 每 epoch 驗證
- 💾 自動保存最佳模型
- 📊 TensorBoard 實時日誌
- 🧪 測試集預測
- 📁 生成 CSV 提交
- ☁️ 自動 Kaggle 上傳

---

### 2. **可視化入口點** (`visualize_da.py`)
獨立的數據增強可視化工具，無需訓練。

```bash
# 模式 1: 單個圖像 + 增強版本
python visualize_da.py --mode single --idx 0 --num-examples 4

# 模式 2: 網格視圖（9 個增強圖像）
python visualize_da.py --mode grid --num-images 9

# 模式 3: 數據集統計（類分布、尺寸等）
python visualize_da.py --mode stats
```

**功能**:
- 👀 展示原始和增強版本
- 📊 類分布分析
- 📐 圖像尺寸統計
- 🎨 可視化對比

---

## 📁 項目結構

```
bcss_segmentation/
│
├── 🎯 entry points
│   ├── train_main.py          ⭐ 訓練入口
│   ├── visualize_da.py        ⭐ 可視化入口
│   └── __main__.py            統一入口
│
├── src/                       模塊化代碼
│   ├── dataset.py             數據加載
│   ├── model.py               U-Net 架構
│   ├── losses.py              損失 + 指標
│   ├── augmentation.py        數據增強
│   └── utils.py               工具函數
│
├── config/                    配置管理
│   └── config.py              所有參數
│
├── 📚 documentation
│   ├── README.md              完整文檔
│   ├── USAGE.md               使用示例
│   ├── PROJECT_STRUCTURE.md   架構圖解
│   └── COMPLETION_SUMMARY.md  重構總結
│
├── 📦 directories (auto-created)
│   ├── ckpt/                  模型檢查點
│   ├── logs/                  TensorBoard + 圖表
│   └── output/                預測結果
│
└── 🔧 setup
    ├── requirements.txt       依賴列表
    ├── setup.sh              自動安裝
    └── check_setup.py        驗證設置
```

---

## 🚀 快速開始

### Step 1: 安裝依賴
```bash
cd bcss_segmentation
pip install -r requirements.txt
```

### Step 2: 驗證設置
```bash
python check_setup.py
```

### Step 3a: 訓練模型
```bash
python train_main.py
```

### Step 3b: 或先探索數據
```bash
python visualize_da.py --mode single
```

---

## 💡 核心功能

### 訓練流程
```
加載數據 → 數據增強 → 初始化模型 → 訓練 100 epochs
  ↓         ↓          ↓              ↓
檢查路徑   Albumentations  U-Net     梯度累積
          (溫和增強)      (注意力)   (有效 batch 96)
  
驗證 → 保存最佳模型 → 測試集預測 → CSV → Kaggle
```

### 模塊關係圖
```
train_main.py
├─ config.py           (配置)
├─ dataset.py          (數據)
├─ model.py            (模型)
├─ losses.py           (損失)
├─ augmentation.py     (增強)
└─ utils.py            (工具)

visualize_da.py
├─ config.py
├─ dataset.py
└─ augmentation.py
```

---

## 📊 可配置參數

所有參數在 `config/config.py` 中，重要配置：

```python
# 數據
BATCH_SIZE = 48              # 物理批大小
ACCUMULATION_STEPS = 4       # 梯度累積步數
NUM_WORKERS = 8              # 數據加載線程

# 模型
N_CLASSES = 3                # 分類類數
USE_ATTENTION = True         # 使用注意力機制

# 訓練
MAX_LR = 1e-3                # 最大學習率
MAX_EPOCHS = 100             # 最大 epoch 數
EARLY_STOPPING_PATIENCE = 10 # 早期停止耐心
GRAD_CLIP = 1.0              # 梯度裁剪
LOSS_WEIGHTS = (0.4, 0.6)    # CE 和 Dice 權重
```

如需調整，直接編輯該文件。

---

## 📈 訓練監控

### 實時 TensorBoard
```bash
# 在新終端運行
tensorboard --logdir=logs
# 訪問 http://localhost:6006
```

### 自動生成的圖表
```
logs/training_history.png  # 訓練/驗證曲線
```

### 查看日誌
```
logs/BCSS_UNet_YYYYMMDD_HHMMSS/  # TensorBoard 事件
```

---

## 🎓 使用示例

### 例 1: 完整流程（推薦）
```bash
# 1. 檢查設置
python check_setup.py

# 2. 探索數據
python visualize_da.py --mode stats

# 3. 訓練模型
python train_main.py

# 4. (自動) 在 Kaggle 上競爭 🏆
```

### 例 2: 快速驗證
```bash
# 跳過 Kaggle 提交
python train_main.py --no-submit
```

### 例 3: 數據分析
```bash
# 查看增強效果
python visualize_da.py --mode single --idx 0 --num-examples 6

# 查看類分布
python visualize_da.py --mode stats

# 查看多個示例
python visualize_da.py --mode grid --num-images 16
```

---

## 🔧 常用命令

| 任務 | 命令 |
|------|------|
| 驗證設置 | `python check_setup.py` |
| 查看幫助 | `python train_main.py --help` |
| 訓練並提交 | `python train_main.py` |
| 訓練但不提交 | `python train_main.py --no-submit` |
| 單圖增強 | `python visualize_da.py --mode single` |
| 網格增強 | `python visualize_da.py --mode grid` |
| 統計信息 | `python visualize_da.py --mode stats` |
| TensorBoard | `tensorboard --logdir=logs` |

---

## 📋 輸出文件

| 位置 | 說明 |
|------|------|
| `ckpt/YYYYMMDD_*/best_model_*.pt` | 最佳模型檢查點 |
| `logs/training_history.png` | 訓練曲線圖 |
| `logs/BCSS_UNet_*/` | TensorBoard 事件 |
| `output/output.csv` | Kaggle 提交文件 |
| `output/augmentation_*.png` | 增強可視化 |
| `output/dataset_statistics.png` | 統計圖表 |

---

## 💾 Kaggle 配置

### 首次設置（一次性）
```bash
# 1. 安裝 Kaggle CLI
pip install kaggle

# 2. 下載 kaggle.json（從 Kaggle 账户）
# 3. 放在 ~/.kaggle/kaggle.json

# 4. 設置權限
chmod 600 ~/.kaggle/kaggle.json
```

### 提交
```bash
# 自動提交
python train_main.py

# 或手動提交
kaggle competitions submit \
  -c lab-4-semantic-segmentation-on-bcss-639003 \
  -f output/output.csv \
  -m "My submission"
```

---

## 🐛 故障排除

### 問題: GPU 內存不足
```python
# config/config.py
BATCH_SIZE = 24
ACCUMULATION_STEPS = 8
```

### 問題: 訓練太慢
```python
# 增加批大小（如果 GPU 允許）
BATCH_SIZE = 96
ACCUMULATION_STEPS = 2

# 或減少 workers
NUM_WORKERS = 4
```

### 問題: 找不到數據
```bash
# 檢查路徑
ls -la ./BCSS/train/ ./BCSS/val/ ./BCSS/test/

# 編輯 config/config.py 中的路徑
```

### 問題: Kaggle 上傳失敗
```bash
# 跳過上傳
python train_main.py --no-submit

# 手動上傳 output/output.csv
```

---

## 📚 文檔

| 文檔 | 內容 |
|------|------|
| `README.md` | 項目概述和功能 |
| `USAGE.md` | 詳細使用示例 |
| `PROJECT_STRUCTURE.md` | 架構圖解 |
| `COMPLETION_SUMMARY.md` | 重構總結 |
| `这个文件` | 快速參考 |

---

## ⚙️ 系統要求

| 項目 | 要求 |
|------|------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.12 |
| GPU | 推薦（CPU 也可以） |
| 磁盤 | ≥ 1 GB（模型+日誌） |
| 內存 | ≥ 8 GB |

---

## 🎉 下一步

1. ✅ **檢查設置**
   ```bash
   python check_setup.py
   ```

2. 🔍 **探索數據**
   ```bash
   python visualize_da.py --mode stats
   ```

3. 🚀 **開始訓練**
   ```bash
   python train_main.py
   ```

4. 📊 **監控進度**
   ```bash
   tensorboard --logdir=logs
   ```

5. 🏆 **上傳 Kaggle**
   ```bash
   # 自動完成或手動上傳 output/output.csv
   ```

---

## 📞 需要幫助？

```bash
# 查看命令幫助
python train_main.py --help
python visualize_da.py --help

# 查看代碼文檔
cat src/model.py       # 查看 U-Net
cat src/losses.py      # 查看損失函數
cat config/config.py   # 查看配置
```

---

**準備好開始了嗎？** 🚀

```bash
python check_setup.py    # 驗證
python train_main.py     # 訓練
```

祝你 Kaggle 競賽成功！ 🏆
