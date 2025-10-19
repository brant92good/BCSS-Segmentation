# 項目結構概覽

```
bcss_segmentation/
│
├── 📁 src/                           # 核心源代碼模塊
│   ├── __init__.py                  # 包初始化
│   ├── dataset.py                   # 🔹 BCSSDataset, BCSSDatasetTest, create_df()
│   ├── model.py                     # 🔹 UNet, DoubleConv, AttentionBlock, Up, Down
│   ├── losses.py                    # 🔹 DiceLoss, pixel_accuracy(), mIoU()
│   ├── augmentation.py              # 🔹 get_train_transforms(), get_val_transforms()
│   └── utils.py                     # 🔹 GPU 設置、檢查點保存/加載、預測
│
├── 📁 config/                        # 配置模塊
│   ├── __init__.py                  # 包初始化
│   └── config.py                    # 🔹 所有超參數和路徑配置
│
├── 📁 ckpt/                         # 模型檢查點（自動創建）
│   └── YYYYMMDD_HHMMSS/            # 時間戳目錄
│       └── best_model_*.pt          # 最佳模型保存
│
├── 📁 logs/                         # TensorBoard 日誌（自動創建）
│   ├── training_history.png         # 訓練曲線圖
│   └── BCSS_UNet_YYYYMMDD_HHMMSS/  # TensorBoard 事件文件
│
├── 📁 output/                       # 預測結果（自動創建）
│   └── output.csv                   # Kaggle 提交文件
│
├── 🎯 train_main.py                 # ⭐ 入口點 1：完整訓練和提交
├── 🎯 visualize_da.py               # ⭐ 入口點 2：數據增強可視化
├── __main__.py                      # 統一入口點
│
├── 📄 requirements.txt               # Python 依賴列表
├── 📄 config.yaml                   # 可選的 YAML 配置
├── 📄 setup.sh                      # 快速設置腳本
├── 📄 README.md                     # 詳細文檔
├── 📄 USAGE.md                      # 使用示例
└── 📄 PROJECT_STRUCTURE.md          # 本文件
```

## 📊 模塊功能對應表

| 模塊 | 主要功能 | 關鍵類/函數 |
|------|---------|-----------|
| **dataset.py** | 數據加載 | BCSSDataset, BCSSDatasetTest, create_df() |
| **model.py** | 模型架構 | UNet, DoubleConv, AttentionBlock, Up, Down |
| **losses.py** | 損失和指標 | DiceLoss, pixel_accuracy(), mIoU() |
| **augmentation.py** | 數據增強 | get_train_transforms(), get_val_transforms() |
| **utils.py** | 工具函數 | setup_gpu(), save_checkpoint(), predict_image() |
| **config.py** | 配置管理 | 所有超參數、路徑配置 |

## 🚀 入口點說明

### 入口點 1: `train_main.py`
```
目的：完整訓練流程
步驟：
  1. 驗證數據路徑
  2. 加載 train/val 數據集
  3. 初始化 U-Net 模型
  4. 執行 100 epochs 訓練（含早期停止）
  5. 每 epoch 驗證
  6. 自動保存最佳模型
  7. 在測試集預測
  8. 生成 CSV
  9. 上傳 Kaggle（可選）

輸出：
  - ckpt/YYYYMMDD_HHMMSS/best_model_*.pt
  - logs/training_history.png
  - output/output.csv
  - logs/BCSS_UNet_*/（TensorBoard）
```

### 入口點 2: `visualize_da.py`
```
目的：數據增強可視化
模式 1 - single：
  - 顯示 1 個原始圖像
  - 顯示 N 個增強版本
  輸出：output/augmentation_example_*.png

模式 2 - grid：
  - 顯示 N 個增強圖像的網格
  輸出：output/augmentation_grid.png

模式 3 - stats：
  - 類分布柱狀圖
  - 圖像尺寸直方圖
  - 數據集分割圓餅圖
  輸出：output/dataset_statistics.png
```

## 🔧 使用命令速查

### 訓練相關
```bash
# 完整訓練 + Kaggle 提交
python train_main.py

# 訓練但不提交
python train_main.py --no-submit

# 通過 __main__.py
python -m bcss_segmentation train
python -m bcss_segmentation train --no-submit
```

### 可視化相關
```bash
# 單個圖像增強（4 個示例）
python visualize_da.py --mode single --idx 0

# 網格（9 個圖像）
python visualize_da.py --mode grid --num-images 9

# 統計信息
python visualize_da.py --mode stats

# 通過 __main__.py
python -m bcss_segmentation visualize --mode single --idx 0
```

## 📈 訓練流程圖

```
train_main.py
    ↓
[驗證數據] → 檢查 BCSS 目錄
    ↓
[加載數據] → create_df() → BCSSDataset → DataLoader
    ↓
[初始化模型] → UNet(n_channels=3, n_classes=3)
    ↓
[訓練循環] 100 epochs
    ├─ [Train Phase]
    │   ├─ Forward pass
    │   ├─ CE Loss + Dice Loss (weighted)
    │   ├─ Backward (gradient accumulation)
    │   ├─ Optimizer step
    │   └─ Metrics (mIoU, Accuracy)
    │
    ├─ [Val Phase]
    │   ├─ Forward pass
    │   ├─ Compute loss & metrics
    │   └─ Check best loss
    │
    └─ [Early Stopping Check]
        └─ If no improvement for 10 epochs → stop
    ↓
[保存最佳模型]
    ↓
[繪製訓練曲線]
    ↓
[測試集預測] → predict_image()
    ↓
[生成 CSV]
    ↓
[Kaggle 提交] (可選)
```

## 📊 可視化流程圖

```
visualize_da.py
    ├─ Mode: single
    │   ├─ Load original image & mask
    │   ├─ Show + 4 augmented versions
    │   └─ Save augmentation_example_*.png
    │
    ├─ Mode: grid
    │   ├─ Load 9 random images
    │   ├─ Apply augmentation to each
    │   └─ Save augmentation_grid.png
    │
    └─ Mode: stats
        ├─ Analyze 100 images
        ├─ Class distribution
        ├─ Image dimensions
        └─ Save dataset_statistics.png
```

## 🔌 模塊間依賴

```
train_main.py
    ├─→ config/config.py          （配置）
    ├─→ src/dataset.py            （數據）
    ├─→ src/model.py              （模型）
    ├─→ src/losses.py             （損失）
    ├─→ src/augmentation.py       （增強）
    └─→ src/utils.py              （工具）

visualize_da.py
    ├─→ config/config.py
    ├─→ src/dataset.py
    └─→ src/augmentation.py
```

## 💾 文件保存位置

| 文件類型 | 位置 | 自動創建 |
|---------|------|--------|
| 模型檢查點 | `ckpt/YYYYMMDD_HHMMSS/` | ✅ |
| 訓練曲線圖 | `logs/training_history.png` | ✅ |
| TensorBoard | `logs/BCSS_UNet_YYYYMMDD_*/` | ✅ |
| 預測結果 | `output/output.csv` | ✅ |
| 增強示例 | `output/augmentation_*.png` | ✅ |
| 統計圖表 | `output/dataset_statistics.png` | ✅ |

## ⚙️ 配置優先級

1. **命令行參數** (最高優先)
   ```bash
   python train_main.py --idx 5
   ```

2. **config/config.py** (次優先)
   ```python
   MAX_EPOCHS = 100
   BATCH_SIZE = 48
   ```

3. **預設值** (最低優先)
   ```python
   argparse 默認值
   ```

## 🎯 快速參考

### 查看幫助
```bash
python train_main.py --help
python visualize_da.py --help
```

### 查看配置
```bash
cat config/config.py
```

### 查看代碼
```bash
cat src/model.py       # UNet 架構
cat src/losses.py      # 損失函數
cat src/dataset.py     # 數據加載
```

### 監控訓練
```bash
# 在另一個終端
tensorboard --logdir=logs
```

訪問 http://localhost:6006

---

**最後更新**: 2024
**版本**: 1.0.0
**狀態**: 生產就緒 ✅
