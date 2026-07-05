```python
"""
山区点云去噪优化模块（场景专属）
底层依赖：Common-LSG constraint_algo 平滑算子
功能：去除植被、飞鸟等离群点，保留真实地形表面
"""
from Common_LSG.constraint_algo import PointCloudSmoother

class TerrainOutlierCleaner(PointCloudSmoother):
    def vegetation_outlier_remove(self, point_cloud, height_threshold=2):
        """
        植被离群点去除：剥离地表以上突出的植被点，还原裸地地形
        """
        terrain_cloud = self.height_baseline_filter(point_cloud, height_threshold)
        return terrain_cloud

    def edge_smooth_optimize(self, terrain_mesh):
        """
        地形边缘顺滑优化：消除阶梯状伪影，适配LSG层级积分特性
        """
        smooth_mesh = self.gradient_smooth_mesh(terrain_mesh, smooth_level=2)
        return smooth_mesh