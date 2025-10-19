# 改進說明文檔 - BCSS 語義分割

## 📝 對話中提到的改進項目

根據聊天記錄分析，主要改進方向：

### 1. 🎯 背景權重 0.2 (Class Weight)

**問題：** 語義分割中，背景像素通常佔 80-90%，導致模型過度關注背景

**解決方案：**
```python
# config/config.py
CLASS_WEIGHTS = [0.2, 1.0, 1.0]  # [background, class1, class2]
```

**作用：**
- 降低背景類別的損失權重
- 強制模型更關注前景類別 (腫瘤組織)
- 改善類別不平衡問題

**實現位置：**
- `src/losses.py` - 新增 `WeightedCrossEntropyLoss`
- `train_main.py` - 使用加權損失函數

---

### 2. 🔧 後處理 (Post-processing)

**對話強調：** "後處理很重要"、"超重要"

**實現的後處理技術：**

#### a) 形態學操作
```python
morphological_cleanup(mask, kernel_size=3)
```
- 閉運算：填補小空洞
- 開運算：移除小噪點

#### b) 移除小物體
```python
remove_small_objects(mask, min_size=50)
```
- 移除面積小於 50 像素的預測
- 減少誤檢

#### c) 填充空洞
```python
fill_holes(mask)
```
- 填補物體內部的空洞
- 使預測更完整

#### d) 測試時增強 (TTA)
```python
test_time_augmentation(model, image, device, num_augments=4)
```
- 對測試圖像做多種變換（翻轉等）
- 對多個預測結果取平均
- 提高預測穩定性

**實現位置：**
- `src/postprocess.py` - 所有後處理函數
- `train_main.py` - 在預測時自動應用

---

### 3. 📊 訓練參數調整

#### a) Epoch 數量: 100 → 40
```python
MAX_EPOCHS = 40  # 原本 100 太多
```
**理由：** 醫學影像數據集較小，40 epochs 通常足夠，避免過擬合

#### b) 學習率: 1e-3 → 3e-4
```python
MAX_LR = 3e-4  # 原本 1e-3 太大
```
**理由：** 較小的學習率讓訓練更穩定，特別是使用 OneCycleLR 時

---

### 4. 🎨 數據增強 2.5x

**對話提到：** "我加了大概 2.5 倍"

**增強的項目：**

原始版本 → 增強版本：
- `ShiftScaleRotate` 機率: 0.3 → 0.5
- 平移限制: 0.05 → 0.1
- 旋轉限制: 10° → 20°
- `RandomBrightnessContrast` 機率: 0.3 → 0.5

**新增增強：**
- `HueSaturationValue` - 色調/飽和度變化
- `ElasticTransform` / `GridDistortion` - 彈性變形
- `CLAHE` - 對比度增強
- `Blur` - 模糊

**實現位置：**
- `src/augmentation.py` - `get_train_transforms()`

---

## 🚀 如何使用改進後的代碼

### 1. 訓練模型

```bash
cd /home/brant/Project/LAB4/bcss_segmentation
python train_main.py
```

### 2. 關鍵配置 (config/config.py)

```python
# 類別權重 (背景 0.2)
CLASS_WEIGHTS = [0.2, 1.0, 1.0]

# 訓練參數
MAX_EPOCHS = 40
MAX_LR = 3e-4

# 後處理參數
POST_PROCESS_MIN_SIZE = 50      # 移除小於 50 像素的物體
POST_PROCESS_KERNEL_SIZE = 3    # 形態學 kernel 大小
POST_PROCESS_FILL_HOLES = True  # 填充空洞
USE_TTA = True                  # 使用測試時增強
```

### 3. 調整參數

如果結果不理想，可以嘗試：

**增加背景抑制：**
```python
CLASS_WEIGHTS = [0.1, 1.0, 1.0]  # 背景權重更低
```

**更積極的後處理：**
```python
POST_PROCESS_MIN_SIZE = 100     # 移除更多小物體
POST_PROCESS_KERNEL_SIZE = 5    # 更大的形態學 kernel
```

**關閉 TTA 加速預測：**
```python
USE_TTA = False  # 預測速度快 4 倍，但可能準確率稍降
```

---

## 📈 預期改進效果

| 改進項目 | 預期效果 | mIoU 提升 |
|---------|---------|----------|
| 類別權重 0.2 | 減少背景誤判 | +2-5% |
| 後處理 | 清理噪點、完整預測 | +3-7% |
| 數據增強 2.5x | 提高泛化能力 | +2-4% |
| 學習率調整 | 更穩定訓練 | +1-2% |
| **總計** | | **+8-18%** |

---

## 🔍 驗證改進

### 訓練時觀察：
```
訓練過程中應該看到：
- Train mIoU 和 Val mIoU 同步提升
- 前景類別的 IoU 明顯改善
- 不會過早過擬合
```

### 預測時觀察：
```
預測結果應該：
- 更少的噪點
- 更完整的物體輪廓
- 更少的小碎片
```

---

## 💡 進階優化建議

如果還需要進一步提升：

1. **更激進的類別權重**
   ```python
   CLASS_WEIGHTS = [0.1, 1.5, 1.5]  # 背景 0.1，前景 1.5
   ```

2. **深度監督 (Deep Supervision)**
   - 在 U-Net 的多個層級添加輔助損失

3. **邊界增強損失**
   - 加入 Boundary Loss 強化邊緣預測

4. **預訓練權重**
   - 使用 ImageNet 預訓練的編碼器

5. **集成學習**
   - 訓練多個模型並平均預測

---

## 📚 相關文件

- `src/losses.py` - 加權交叉熵損失
- `src/postprocess.py` - 後處理函數
- `src/augmentation.py` - 增強數據增強
- `config/config.py` - 所有配置參數
- `train_main.py` - 主訓練腳本

---

## ❓ 疑難排解

**Q: 訓練很慢？**
A: 關閉 TTA (`USE_TTA = False`)，減少 `ACCUMULATION_STEPS`

**Q: 記憶體不足？**
A: 降低 `BATCH_SIZE` 或增加 `ACCUMULATION_STEPS`

**Q: 過擬合？**
A: 增加數據增強機率，降低 `MAX_EPOCHS`

**Q: 欠擬合？**
A: 增加 `MAX_EPOCHS`，提高 `MAX_LR`
