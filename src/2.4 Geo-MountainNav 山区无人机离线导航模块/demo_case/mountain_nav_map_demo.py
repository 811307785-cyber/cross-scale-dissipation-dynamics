"""
山区离线导航建图Demo
"""
from Geo_MountainNav.hardware_adapt.lidar_vision_parser import LidarVisionParser
from Geo_MountainNav.noise_optimize.pointcloud_outlier_clean import TerrainOutlierCleaner
from Geo_MountainNav.scene_constraint.terrain_gradient_constraint import TerrainGradientConstraint
from Common_LSG.slice_rebuild import SectionRebuilder

def mountain_map_pipeline(las_path, output_path):
    parser = LidarVisionParser()
    terrain_data = parser.load_las_pointcloud(las_path)
    
    cleaner = TerrainOutlierCleaner()
    cleaned_stack = cleaner.vegetation_outlier_remove(terrain_data.section_stack)
    
    constraint = TerrainGradientConstraint()
    constrained_stack = [constraint.slope_boundary_constrain(s) for s in cleaned_stack]
    
    rebuilder = SectionRebuilder()
    terrain_model = rebuilder.non_orthogonal_rebuild(constrained_stack, terrain_data.spacing)
    
    print("山区高精地形图构建完成，支持离线导航路径规划")
    return terrain_model

if __name__ == "__main__":
    mountain_map_pipeline("./mountain.las", "./output/mountain_map.stl")