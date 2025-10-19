# BCSS Breast Cancer Semantic Segmentation

醫學影像語義分割專案 - 針對乳腺癌組織切片進行多類別分割

## 📊 專案概述

- **任務**: 3 類語義分割 (背景, 類別1, 類別2)
- **圖像大小**: 224×224
- **模型**: U-Net with Attention
- **後處理**: 形態學清理 + TTA


## 🚀 快速開始

### 1. 環境設置

```bash
# 創建 conda 環境
conda create -f lab4.yml
conda activate bcss

```

### 2. 訓練模型

```bash
python train_main.py
```

### 3. 僅預測 (使用已訓練模型)

```bash
python predict_only.py --model ckpt/20251019_173024/best_model_loss-0.2106_mIoU-0.512.pt
```

## 📁 專案結構

```
bcss_segmentation/
├── config/
│   └── config.py              # 所有配置參數
├── src/
│   ├── augmentation.py        # 數據增強 (2.5x)
│   ├── dataset.py             # 數據載入
│   ├── losses.py              # 損失函數 (含加權CE)
│   ├── model.py               # U-Net 模型
│   ├── postprocess.py         # 後處理函數
│   └── utils.py               # 工具函數
├── train_main.py              # 主訓練腳本
├── predict_only.py            # 僅預測腳本
├── analyze_object_sizes.py    # 物體大小分析
└── BCSS/                      # 數據集目錄
    ├── train/
    ├── train_mask/
    ├── val/
    ├── val_mask/
    └── test/
```

## 📈 監控訓練

```bash
# 使用 TensorBoard
tensorboard --logdir logs --port 6006
```

## 🔍 分析工具

### 分析物體大小分布
```bash
python analyze_object_sizes.py ./BCSS/train_mask/
```

輸出:
- 物體大小統計
- 分佈圖
- 建議的 `min_size` 參數

## 📚 文檔

- `IMPROVEMENTS.md` - 詳細改進說明
- `PERFORMANCE_TARGET.md` - 性能目標與分析
- `POSTPROCESS_TUNING.md` - 後處理參數調優指南

## 📊 輸出

訓練完成後會生成:
- `ckpt/<timestamp>/best_model_*.pt` - 最佳模型
- `logs/training_history.png` - 訓練曲線
- `output/output.csv` - 預測結果

## 🙏 致謝

基於 U-Net 架構，整合多種最佳實踐

## 📄 授權

請遵守數據集的使用條款
