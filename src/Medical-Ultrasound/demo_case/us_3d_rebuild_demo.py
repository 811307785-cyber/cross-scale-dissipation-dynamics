"""
超声三维重建Demo
"""
from Medical_Ultrasound.hardware_adapt.us_signal_parser import USSignalParser
from Medical_Ultrasound.noise_optimize.us_speckle_denoise import USSpeckleDenoise
from Medical_Ultrasound.scene_constraint.us_acoustic_constraint import USAcousticConstraint
from Common_LSG.slice_rebuild import SectionRebuilder

def us_rebuild_pipeline(us_dicom_path, output_path):
    parser = USSignalParser()
    us_data = parser.load_us_dicom(us_dicom_path)
    
    denoiser = USSpeckleDenoise()
    denoised_stack = [denoiser.speckle_noise_suppress(s) for s in us_data.section_stack]
    
    constraint = USAcousticConstraint()
    constrained_stack = constraint.depth_attenuation_compensate(denoised_stack)
    
    rebuilder = SectionRebuilder()
    volume = rebuilder.non_orthogonal_rebuild(constrained_stack, us_data.spacing)
    
    print("超声三维重建完成")
    return volume

if __name__ == "__main__":
    us_rebuild_pipeline("./sample_us.dcm", "./output/us_model.stl")