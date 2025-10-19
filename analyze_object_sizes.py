#!/usr/bin/env python3
"""
分析數據集中物體大小分布，找出最佳 min_size 參數
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_mask_sizes(mask_dir, sample_size=100):
    """
    分析 mask 中物體大小的分布
    
    Args:
        mask_dir: mask 目錄路徑
        sample_size: 採樣數量
    """
    print(f"🔍 分析 mask 目錄: {mask_dir}")
    
    if not os.path.exists(mask_dir):
        print(f"❌ 目錄不存在: {mask_dir}")
        return
    
    # 收集所有 mask 文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    if len(mask_files) == 0:
        print(f"❌ 沒有找到 mask 文件")
        return
    
    # 限制樣本數量
    if len(mask_files) > sample_size:
        mask_files = np.random.choice(mask_files, sample_size, replace=False)
    
    print(f"📊 分析 {len(mask_files)} 個 mask 文件...")
    
    # 統計數據
    object_sizes = defaultdict(list)  # {class_id: [size1, size2, ...]}
    total_objects = defaultdict(int)
    
    for filename in tqdm(mask_files, desc="Processing"):
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        # 對每個類別分析
        for class_id in np.unique(mask):
            if class_id == 0:  # 跳過背景
                continue
            
            # 創建二值化 mask
            binary_mask = (mask == class_id).astype(np.uint8)
            
            # 找連通組件
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            
            # 收集每個物體的大小
            for i in range(1, num_labels):  # 跳過背景 (0)
                size = stats[i, cv2.CC_STAT_AREA]
                object_sizes[class_id].append(size)
                total_objects[class_id] += 1
    
    # 生成報告
    print("\n" + "="*80)
    print("📊 物體大小分析報告")
    print("="*80)
    
    for class_id in sorted(object_sizes.keys()):
        sizes = np.array(object_sizes[class_id])
        
        print(f"\n類別 {class_id}:")
        print(f"  總物體數量: {len(sizes)}")
        print(f"  最小大小: {sizes.min():.1f} 像素")
        print(f"  最大大小: {sizes.max():.1f} 像素")
        print(f"  平均大小: {sizes.mean():.1f} 像素")
        print(f"  中位數: {np.median(sizes):.1f} 像素")
        print(f"  標準差: {sizes.std():.1f} 像素")
        print(f"\n  百分位數:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(sizes, p)
            print(f"    {p}%: {val:.1f} 像素")
        
        # 統計不同 min_size 會移除多少物體
        print(f"\n  不同 min_size 閾值影響:")
        for threshold in [10, 20, 30, 50, 100, 200]:
            removed = np.sum(sizes < threshold)
            percentage = (removed / len(sizes)) * 100
            print(f"    min_size={threshold:3d}: 移除 {removed:4d} ({percentage:5.2f}%)")
    
    # 繪製分佈圖
    print("\n" + "="*80)
    print("📈 生成分佈圖...")
    
    fig, axes = plt.subplots(1, len(object_sizes), figsize=(6*len(object_sizes), 5))
    
    if len(object_sizes) == 1:
        axes = [axes]
    
    for idx, (class_id, sizes) in enumerate(sorted(object_sizes.items())):
        ax = axes[idx]
        
        # 直方圖
        ax.hist(sizes, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('物體大小 (像素)', fontsize=12)
        ax.set_ylabel('數量', fontsize=12)
        ax.set_title(f'類別 {class_id} - 物體大小分佈', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 標記常見的 min_size 閾值
        for threshold in [50, 100, 200]:
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'min_size={threshold}', alpha=0.7)
        
        ax.legend()
    
    plt.tight_layout()
    output_path = 'object_size_distribution.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ 分佈圖已保存: {output_path}")
    
    # 建議
    print("\n" + "="*80)
    print("💡 建議")
    print("="*80)
    
    all_sizes = []
    for sizes in object_sizes.values():
        all_sizes.extend(sizes)
    all_sizes = np.array(all_sizes)
    
    p1 = np.percentile(all_sizes, 1)
    p5 = np.percentile(all_sizes, 5)
    p10 = np.percentile(all_sizes, 10)
    
    print(f"\n基於統計分析:")
    print(f"  • 1% 的物體 < {p1:.0f} 像素 (可能是噪點)")
    print(f"  • 5% 的物體 < {p5:.0f} 像素")
    print(f"  • 10% 的物體 < {p10:.0f} 像素")
    
    print(f"\n建議的 min_size 設定:")
    if p1 < 20:
        print(f"  🔹 保守 (保留更多物體): min_size = {int(p1)}-{int(p5)}")
    else:
        print(f"  🔹 保守 (保留更多物體): min_size = 10-20")
    
    print(f"  🔹 平衡 (推薦): min_size = {int(p5)}-{int(p10)}")
    print(f"  🔹 激進 (移除更多噪點): min_size = {int(p10)}-100")
    
    print(f"\n當前設定 (min_size=50):")
    removed_current = np.sum(all_sizes < 50)
    percentage_current = (removed_current / len(all_sizes)) * 100
    print(f"  會移除 {removed_current} 個物體 ({percentage_current:.2f}%)")
    
    if percentage_current < 5:
        print(f"  ✅ 合理：只移除少量小物體")
    elif percentage_current < 15:
        print(f"  ⚠️  適中：移除一些物體，可能合理")
    else:
        print(f"  ❌ 過大：會移除太多物體，建議降低")


if __name__ == "__main__":
    import sys
    
    # 可以從命令列指定目錄
    if len(sys.argv) > 1:
        mask_dir = sys.argv[1]
    else:
        # 預設目錄
        mask_dir = './BCSS/train_mask/'
    
    analyze_mask_sizes(mask_dir, sample_size=200)
