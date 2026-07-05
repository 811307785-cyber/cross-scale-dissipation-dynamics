"""
超声斑点噪声抑制模块（场景专属）
底层依赖：Common-LSG constraint_algo 平滑算子
功能：抑制超声固有斑点噪声，保留组织纹理细节
"""
from Common_LSG.constraint_algo import AdaptiveSmoother

class USSpeckleDenoise(AdaptiveSmoother):
    def speckle_noise_suppress(self, section_matrix, window_size=3):
        """
        自适应Lee滤波斑点噪声抑制
        """
        denoised_matrix = self.adaptive_lee_filter(section_matrix, window_size)
        return denoised_matrix

    def reverberation_artifact_remove(self, section_stack):
        """
        混响伪影消除：去除多次反射产生的重复层状伪影
        """
        cleaned_stack = self.periodic_artifact_filter(section_stack, direction="depth")
        return cleaned_stack