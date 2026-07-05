
##### scene_constraint/us_acoustic_constraint.py
```python
"""
超声声学约束模块（场景专属）
底层依赖：Common-LSG constraint_algo 基类
功能：基于声速、声学阻抗约束组织界面，适配超声传播物理特性
"""
from Common_LSG.constraint_algo import BaseSectionConstraint

class USAcousticConstraint(BaseSectionConstraint):
    # 人体组织标准声速约束（m/s）
    TISSUE_SOUND_SPEED = {
        "water": 1540,
        "soft_tissue": 1580,
        "bone": 4080,
        "air": 330
    }

    def acoustic_interface_constrain(self, section_matrix, tissue_sound_speed=1580):
        """
        声学界面约束：基于声阻抗差异强化组织边界，提升分层重建精度
        """
        impedance_matrix = section_matrix * tissue_sound_speed
        constrained_matrix = self.gradient_boundary_constrain(impedance_matrix, threshold=0.15)
        return constrained_matrix

    def depth_attenuation_compensate(self, section_stack, frequency=3.5):
        """
        深度衰减补偿：按超声传播深度修正灰度衰减，适配层级积分
        """
        compensated_stack = self.depth_gradient_correct(section_stack, attenuation_coeff=0.5*frequency)
        return compensated_stack