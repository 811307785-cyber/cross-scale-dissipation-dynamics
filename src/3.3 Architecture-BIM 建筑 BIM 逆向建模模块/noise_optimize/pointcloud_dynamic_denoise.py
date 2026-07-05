"""
建筑点云动态去噪模块（场景专属）
底层依赖：Common-LSG constraint_algo 点云平滑算子
功能：移除点云中行人、车辆等动态物体，保留建筑静态结构
"""
from Common_LSG.constraint_algo import PointCloudSmoother

class BuildingPointCloudDenoise(PointCloudSmoother):
    def dynamic_object_remove(self, point_cloud):
        """动态物体离群点移除，保留静态建筑结构"""
        static_cloud = self.static_baseline_filter(point_cloud)
        return static_cloud

    def building_edge_optimize(self, building_mesh):
        """建筑边缘顺滑优化，消除阶梯状伪影"""
        smooth_mesh = self.gradient_smooth_mesh(building_mesh, smooth_level=1)
        return smooth_mesh