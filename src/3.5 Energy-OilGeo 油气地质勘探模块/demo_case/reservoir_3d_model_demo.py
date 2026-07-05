"""
油气藏三维建模Demo
"""
from Energy_OilGeo.hardware_adapt.seismic_logging_parser import SeismicLoggingParser
from Energy_OilGeo.noise_optimize.seismic_noise_suppress import SeismicNoiseSuppress
from Energy_OilGeo.scene_constraint.stratum_deposition_constraint import StratumDepositionConstraint
from Common_LSG.slice_rebuild import SectionRebuilder

def reservoir_model_pipeline(segy_path, output_path):
    parser = SeismicLoggingParser()
    seismic_data = parser.load_segy_section(segy_path)
    
    denoiser = SeismicNoiseSuppress()
    cleaned_stack = [denoiser.surface_wave_remove(s) for s in seismic_data.section_stack]
    
    constraint = StratumDepositionConstraint()
    constrained_stack = constraint.stratum_layer_constrain(cleaned_stack)
    
    rebuilder = SectionRebuilder()
    reservoir_model = rebuilder.non_orthogonal_rebuild(constrained_stack, seismic_data.spacing)
    
    print("油气藏三维地质建模完成")
    return reservoir_model

if __name__ == "__main__":
    reservoir_model_pipeline("./seismic.segy", "./output/reservoir_model.stl")