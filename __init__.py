"""
@title: fumarole_plume_model
    A collection of tools for the analysis (from infrared thermal images) and
    synthesis of fumarole plume motion.
@author: D. E. Jessop and A. Klein
@date: 2025-09-11
"""
from fumarole_plume_model.numerical_plume_model import (derivs,
                                                        entrainment_vel,
                                                        wind_profile,
                                                        density_atm,
                                                        density_fume,
                                                        heat_capacity,
                                                        bulk_gas_constant,
                                                        temperature_fume,
                                                        produce_Gm,
                                                        objective_fn,
                                                        parallel_job,
                                                        solve_system,
                                                        do_plots)
from fumarole_plume_model.bent_plume_analyser import (centroid_posn,
                                                      plume_trajectory,
                                                      gaussian_profile,
                                                      calc_scale_factor,
                                                      image_extent,
                                                      pixel_to_world,
                                                      world_to_pixel,
                                                      show_scaled_image,
                                                      open_plot_expt_image,
                                                      dist_along_path,
                                                      plume_angle,
                                                      initial_guess_at_axis,
                                                      rotate_image,
                                                      true_location_width,
                                                      path_from_smoothed_theta,
                                                      wacky_value,
                                                      extract_temperatures,
                                                      read_params_file,
                                                      save_params_file,
                                                      _line,
                                                      plume_analysis)
from fumarole_plume_model.expt_plume_model import (derivs as expt_derivs,
                                                   integrator,
                                                   integrator2, 
                                                   objective_fn as expt_objf, 
                                                   objective_fn2 as expt_objf2,
                                                   objective_fn3 as expt_objf3,
                                                   wind,
                                                   load_ics_parameters,
                                                   load_expt_data)
from fumarole_plume_model._version import __version__
