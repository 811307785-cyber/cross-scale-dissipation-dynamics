"""
超声数据适配接口（场景专属）
底层依赖：Common-LSG io_utils 通用IO工具
功能：解析超声DICOM、原始射频信号，转换为标准截面格式
"""
import numpy as np
from Common_LSG.io_utils import SectionDataStandard

class USSignalParser(SectionDataStandard):
    SUPPORTED_MODALITY = "US"

    def load_us_dicom(self, dicom_file_path):
        """加载超声DICOM图像序列"""
        try:
            import pydicom
            ds = pydicom.dcmread(dicom_file_path)
        except ImportError:
            raise ImportError("需安装pydicom依赖")
        
        pixel_array = ds.pixel_array
        # 超声多帧序列处理
        if len(pixel_array.shape) == 3:
            section_stack = pixel_array
        else:
            section_stack = np.expand_dims(pixel_array, axis=0)
        
        standard_data = self.format_section_stack(
            section_stack,
            spacing=[ds.PixelSpacing[0], ds.PixelSpacing[1], 1.0],
            origin=ds.ImagePositionPatient if hasattr(ds, "ImagePositionPatient") else [0,0,0]
        )
        return standard_data