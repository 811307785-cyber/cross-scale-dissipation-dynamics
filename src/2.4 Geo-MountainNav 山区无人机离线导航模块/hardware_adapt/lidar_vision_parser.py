"""
无人机LiDAR/视觉数据适配接口
底层依赖：Common-LSG io_utils 通用IO工具
功能：解析激光点云、视觉航片，转换为LSG标准截面输入
"""
import numpy as np
from Common_LSG.io_utils import SectionDataStandard

class LidarVisionParser(SectionDataStandard):
    def load_las_pointcloud(self, las_file_path):
        """加载LAS格式LiDAR点云"""
        try:
            import laspy
        except ImportError:
            raise ImportError("需安装laspy依赖：pip install laspy")
        
        las = laspy.read(las_file_path)
        points = np.vstack([las.x, las.y, las.z]).T
        
        # 点云转截面堆叠格式
        section_stack = self.pointcloud_to_section_stack(points, resolution=0.5)
        standard_data = self.format_section_stack(section_stack, spacing=[0.5, 0.5, 0.5])
        return standard_data