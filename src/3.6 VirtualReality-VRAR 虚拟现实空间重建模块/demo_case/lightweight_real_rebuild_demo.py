"""
VR轻量化实时空间重建Demo
优化目标：低延迟、低算力，适配VR头显嵌入式芯片
"""
from VirtualReality_VRAR.hardware_adapt.rgbd_vr_parser import RGBDVRParser
from VirtualReality_VRAR.noise_optimize.depth_hole_repair import DepthHoleRepair
from VirtualReality_VRAR.scene_constraint.spatial_plane_constraint import SpatialPlaneConstraint
from Common_LSG.slice_rebuild import SectionRebuilder

def vr_rebuild_pipeline(depth_array, intrinsic):
    parser = RGBDVRParser()
    depth_data = parser.load_vr_depth_frame(depth_array, intrinsic)
    
    repair = DepthHoleRepair()
    repaired = repair.depth_hole_fill(depth_data.section_stack[0])
    
    constraint = SpatialPlaneConstraint()
    optimized = constraint.depth_hierarchy_optimize(repaired, level_num=8)
    
    rebuilder = SectionRebuilder(lightweight_mode=True)
    space_model = rebuilder.non_orthogonal_rebuild([optimized], depth_data.spacing)
    
    print("VR空间轻量化重建完成，支持实时渲染")
    return space_model

if __name__ == "__main__":
    import numpy as np
    demo_depth = np.random.rand(480, 640).astype(np.float32)
    demo_intrinsic = np.eye(3) * 500
    vr_rebuild_pipeline(demo_depth, demo_intrinsic)