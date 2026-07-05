"""
工业CT伪影优化模块（场景专属）
底层依赖：Common-LSG constraint_algo 平滑算子
功能：抑制高对比度工件的射束硬化、散射伪影
"""
from Common_LSG.constraint_algo import GradientSmoother

class IndustrialCTArtifactOptimize(GradientSmoother):
    def high_contrast_artifact_remove(self, section_matrix):
        """高对比度金属工件射束硬化伪影消除"""
        repaired_matrix = self.hardening_artifact_correct(section_matrix)
        return repaired_matrix

    def scatter_noise_suppress(self, section_stack):
        """散射噪声全局抑制，保留微小缺陷细节"""
        suppressed_stack = self.scatter_gray_balance(section_stack)
        return suppressed_stack