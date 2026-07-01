python
"""
层级截面几何(LSG) - 离线导航通用地形匹配内核示例
Layered Section Geometry - General Terrain Matching Kernel for Offline Navigation
对应理论：层级采样、截面梯度、四级分类、地形指纹匹配
Corresponding Theory: Hierarchical Sampling, Section Gradient, Four-Level Classification, Terrain Fingerprint Matching
"""

import numpy as np
import rasterio

# ==================================================
# 1. 加载DEM地形数据，构建层级截面采样 (对应层级采样定义)
# 1. Load DEM data, build hierarchical section sampling
# ==================================================
def load_dem_sections(dem_path, slice_interval=10):
    """
    输入: DEM栅格文件路径, 高程切片间隔(米)
    输出: 地形三维数组 + 高程层列表 + 平面坐标仿射参数
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        nodata = src.nodata
    
    # 屏蔽无效值
    dem[dem == nodata] = np.nan
    
    # 生成等间隔高程切片层级
    min_h = np.nanmin(dem)
    max_h = np.nanmax(dem)
    height_levels = np.arange(min_h, max_h, slice_interval)
    
    return dem, height_levels, transform

# ==================================================
# 2. 计算每层截面特征与梯度序列 (对应截面梯度定义)
# 2. Calculate per-layer section features and gradient sequence
# ==================================================
def calc_section_gradient(dem, height_levels):
    """
    逐高程层计算闭合等高线的等效面积、周长，推导一/二阶梯度
    输出: 每层面积序列A、周长序列C、一阶梯度g1、二阶梯度g2
    """
    area_seq = []
    perimeter_seq = []
    
    for h in height_levels:
        # 提取当前高程以上的连通区域
        mask = dem >= h
        if not np.any(mask):
            area_seq.append(0)
            perimeter_seq.append(0)
            continue
        
        # 等效截面面积(像素数转实际面积)
        area = np.sum(mask)
        area_seq.append(area)
        
        # 等效截面周长(边界像素数)
        boundary = mask & ~np.roll(mask, 1, axis=0) | ~np.roll(mask, 1, axis=1)
        perimeter = np.sum(boundary)
        perimeter_seq.append(perimeter)
    
    # 转为numpy数组
    area_seq = np.array(area_seq)
    perimeter_seq = np.array(perimeter_seq)
    dh = height_levels[1] - height_levels[0]
    
    # 一阶梯度
    g1 = np.gradient(area_seq, dh)
    # 二阶梯度
    g2 = np.gradient(g1, dh)
    
    return area_seq, perimeter_seq, g1, g2

# ==================================================
# 3. 构建地形指纹特征向量 (对应地形指纹定义)
# 3. Build terrain fingerprint feature vector
# ==================================================
def build_terrain_fingerprint(area_seq, g1, g2, window_size=5):
    """
    滑动窗口提取多层复合梯度特征，构成唯一地形指纹
    输出: 指纹特征序列，用于匹配定位
    """
    n = len(area_seq)
    fingerprints = []
    
    for i in range(window_size, n - window_size):
        # 窗口内复合特征：面积、一阶梯度、二阶梯度拼接
        feature = np.concatenate([
            area_seq[i-window_size:i+window_size+1],
            g1[i-window_size:i+window_size+1],
            g2[i-window_size:i+window_size+1]
        ])
        fingerprints.append(feature)
    
    return np.array(fingerprints)

# ==================================================
# 4. 地形特征匹配：观测指纹匹配全局指纹库 (基础定位)
# 4. Terrain feature matching: observation fingerprint vs global fingerprint database
# ==================================================
def match_terrain_position(obs_fingerprint, fingerprint_db, candidate_coords):
    """
    输入: 实时观测地形指纹, 全局指纹库, 指纹对应平面坐标
    输出: 匹配最优的平面坐标 (绝对定位结果)
    """
    # 计算最小残差
    residuals = np.sum(np.abs(fingerprint_db - obs_fingerprint), axis=1)
    best_idx = np.argmin(residuals)
    
    return candidate_coords[best_idx], residuals[best_idx]

# ==================================================
# 5. 四级梯度地形分类 (对应G0-G3四级体系)
# 5. Four-level gradient terrain classification
# ==================================================
def classify_terrain_type(g1_avg, g2_avg):
    """
    根据平均梯度判定地形区段所属四级类型
    """
    if abs(g1_avg) < 1e-3:
        return "G0 恒定梯度类（平缓台地/谷底）"
    elif abs(g2_avg) < 1e-3:
        return "G2 线性梯度类（均匀缓坡/山谷边坡）"
    elif abs(g2_avg) > 1e-1 and abs(g1_avg) > 1e-1:
        return "G3 剧变梯度类（陡崖/尖峰/峡谷）"
    else:
        return "G1 缓变非线性类（圆润山体/丘陵）"

# ==================================================
# 主程序示例
# ==================================================
if __name__ == '__main__':
    # 替换为你的DEM文件路径
    DEM_PATH = './mountain_dem.tif'
    
    # 1. 加载地形，生成层级截面
    dem, h_levels, transform = load_dem_sections(DEM_PATH, slice_interval=10)
    print(f"地形尺寸 / Terrain size: {dem.shape}, 高程层数 / Elevation layers: {len(h_levels)}")
    
    # 2. 计算截面梯度序列
    area_seq, peri_seq, g1, g2 = calc_section_gradient(dem, h_levels)
    
    # 3. 构建全局地形指纹库
    fp_db = build_terrain_fingerprint(area_seq, g1, g2)
    print(f"全局指纹库规模 / Global fingerprint database size: {fp_db.shape}")
    
    # 4. 模拟观测片段匹配定位（取中间窗口作为观测值）
    mid = len(area_seq) // 2
    obs_fp = np.concatenate([
        area_seq[mid-5:mid+6],
        g1[mid-5:mid+6],
        g2[mid-5:mid+6]
    ])
    
    # 简化：坐标候选集（实际使用绑定每个指纹对应平面坐标）
    coords = np.arange(len(fp_db))
    best_pos, residual = match_terrain_position(obs_fp, fp_db, coords)
    print(f"最优匹配位置索引 / Best matching index: {best_pos}, 匹配残差 / Matching residual: {residual:.2f}")
    
    # 5. 地形类型分类
    terrain_type = classify_terrain_type(np.mean(g1), np.mean(g2))
    print(f"当前区段地形类型 / Current terrain type: {terrain_type}")
