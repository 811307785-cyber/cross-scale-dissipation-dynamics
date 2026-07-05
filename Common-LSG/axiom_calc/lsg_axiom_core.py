"""
LSG 层级截面几何 - 公理核心计算库
对应四条本源公理的数学实现：
1. 三维空间本底守恒 -> 截面映射与空间基底
2. 空间无限可分 -> 层级梯度积分
3. 正负维度双层互易 -> 维度互易变换矩阵
4. 单一本源对偶力 -> 梯度差与平衡约束
"""

import numpy as np


def section_mapping(points_3d, plane_normal, plane_origin):
    """
    公理1：三维空间截面映射
    将三维空间点投影到任意截面平面，得到二维截面坐标
    :param points_3d: 三维点集 (N, 3)
    :param plane_normal: 截面法向量 (3,)
    :param plane_origin: 截面原点 (3,)
    :return: 截面二维坐标 (N, 2)
    """
    # 法向量单位化
    normal = plane_normal / np.linalg.norm(plane_normal)
    # 构建截面局部坐标系
    if abs(normal[2]) < 0.9:
        u_axis = np.cross(normal, np.array([0, 0, 1]))
    else:
        u_axis = np.cross(normal, np.array([1, 0, 0]))
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    # 点到截面平面的投影
    vec = points_3d - plane_origin
    proj_dist = np.dot(vec, normal)
    proj_points = points_3d - np.outer(proj_dist, normal)

    # 转换为截面局部二维坐标
    u = np.dot(proj_points - plane_origin, u_axis)
    v = np.dot(proj_points - plane_origin, v_axis)
    return np.column_stack((u, v))


def layer_gradient_integral(section_list, layer_thickness):
    """
    公理2：层级梯度积分
    多层截面沿梯度方向积分，还原三维连续场分布
    :param section_list: 多层截面数据列表，每层为二维数值矩阵
    :param layer_thickness: 层间厚度
    :return: 三维体数据矩阵
    """
    layer_count = len(section_list)
    if layer_count == 0:
        return np.array([])
    
    base_shape = section_list[0].shape
    volume = np.zeros((base_shape[0], base_shape[1], layer_count))
    
    for i, sec in enumerate(section_list):
        volume[:, :, i] = sec
    
    # 梯度方向连续插值平滑（空间无限可分的连续化实现）
    if layer_count > 1:
        z = np.arange(layer_count) * layer_thickness
        from scipy.interpolate import interp1d
        for i in range(base_shape[0]):
            for j in range(base_shape[1]):
                f = interp1d(z, volume[i, j, :], kind='linear', fill_value="extrapolate")
                volume[i, j, :] = f(z)
    
    return volume


def dimension_reciprocal_transform(entity_field, mode="positive_to_negative"):
    """
    公理3：正负维度双层互易变换
    正维度实体结构 <-> 负维度空间场域 双向转换
    :param entity_field: 输入场矩阵
    :param mode: 变换方向 positive_to_negative / negative_to_positive
    :return: 变换后对偶场
    """
    if mode == "positive_to_negative":
        # 实体转空间场：实体边界向外梯度扩散
        max_val = np.max(entity_field) if np.max(entity_field) != 0 else 1
        inverse_field = max_val - entity_field
        # 梯度平滑模拟空间场扩散
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(inverse_field, sigma=1.0)
    
    elif mode == "negative_to_positive":
        # 空间场转实体：场强阈值收敛为实体边界
        threshold = np.mean(entity_field) + np.std(entity_field) * 0.5
        entity = np.where(entity_field > threshold, 1.0, 0.0)
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(entity).astype(float)
    
    else:
        raise ValueError("mode must be positive_to_negative or negative_to_positive")


def dual_force_gradient(field_data):
    """
    公理4：对偶力梯度计算
    计算场分布的吸引-排斥对偶力梯度差
    :param field_data: 输入标量场
    :return: 梯度矢量场, 梯度强度标量场
    """
    grad_y, grad_x = np.gradient(field_data)
    grad_vector = np.stack((grad_x, grad_y), axis=-1)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return grad_vector, grad_magnitude
