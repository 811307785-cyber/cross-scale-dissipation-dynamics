"""
地形梯度约束模块（场景专属）
底层依赖：Common-LSG constraint_algo 基类
功能：基于山体自然坡度约束地形重建，消除断崖式伪影
"""
from Common_LSG.constraint_algo import BaseSectionConstraint

class TerrainGradientConstraint(BaseSectionConstraint):
    # 山地地形合理坡度范围约束（0-90度）
    MAX_MOUNTAIN_SLOPE = 75  # 最大自然山体坡度
    MIN_TERRAIN_GRADIENT = 0

    def slope_boundary_constrain(self, terrain_section, max_slope_degree=75):
        """
        坡度边界约束：限制单截面地形梯度突变，符合自然山体形态
        """
        constrained_section = self.gradient_limit_constrain(
            terrain_section, 
            max_gradient=np.tan(np.radians(max_slope_degree))
        )
        return constrained_section

    def elevation_layer_constrain(self, section_stack, elevation_range):
        """
        高程分层约束：按海拔分层重建，适配山区大落差地形
        """
        min_elev, max_elev = elevation_range
        layered_stack = self.layer_interval_constrain(section_stack, min_elev, max_elev, layer_num=20)
        return layered_stack