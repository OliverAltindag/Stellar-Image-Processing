import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
import os
import glob
import helper_functions as h
import main_functions as mf

# Very specific helper fucntions

def sort_and_align_files(science_folder_path, star_coords, background_coords):
    '''
    '''
    search_pattern = os.path.join(science_folder_path, "**", "fdb_*.fit")
    fdb_science_files = glob.glob(search_pattern, recursive=True)

    filter_files = {
        'Visual': [],
        'Blue': [],
        'Red': []
    }

    for path in fdb_science_files:
        filter_name = fits.getheader(path)["FILTER"].strip()
        if filter_name in filter_files:
            filter_files[filter_name].append(path)

    all_shifts = {}
    for filter_name, file_list in filter_files.items():
        current_shift_list = []
        image_path_ref = file_list[0] # Use the first file as the reference
        for image_path in file_list:
            x_y_shifts = mf.centroiding(image_path, image_path_ref, star_coords, background_coords)
            if x_y_shifts is None:
                print(f"User aborted alignment for filter: {filter_name}. Check star/background boxes.")
                current_shift_list = [] # Clear any shifts
                break
            current_shift_list.append(x_y_shifts)    
        all_shifts[filter_name] = current_shift_list
    return filter_files, all_shifts

def stack_all_filters(folder_path, filter_files, all_shifts, pad_val):
    '''
    '''
    for filter_name in filter_files.keys():
        file_list = filter_files.get(filter_name, [])
        shifts_list = all_shifts.get(filter_name, [])
        save_path = os.path.join(folder_path, f'master_stack_{filter_name.lower()}.fit')
        x_shifts = [s[0] for s in shifts_list]
        y_shifts = [s[1] for s in shifts_list]
        mf.shifting(file_list, x_shifts, y_shifts, pad_val, save_path)
    return
    
def align_and_stack_folder(folder_path, star_coords, bg_coords, pad_val):
    '''
    '''
    filter_files, all_shifts = sort_and_align_files(folder_path, star_coords, bg_coords)
    stack_all_filters(folder_path, filter_files, all_shifts, pad_val)
    return

# The actual image reduction and stacking is done

def reduction(data_folder_path, science_images_folder):
    '''
    '''
    standard_images_subfolder = "standard"
    standard_folder_path = os.path.join(data_folder_path, standard_images_subfolder)
    science_folder_path = os.path.join(data_folder_path, science_images_folder)

    bias_files = glob.glob(os.path.join(data_folder_path, 'calibration/biasframes', '*.fit'))
    dark_files = glob.glob(os.path.join(data_folder_path, 'calibration/darks', '*.fit'))
    visual_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/visual', '*.fit'))
    blue_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/blue', '*.fit'))
    red_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/red', '*.fit'))
    
    master_bias_path = os.path.join(data_folder_path, 'calibration/biasframes/master_bias.fit')
    mf.master_bias(bias_files, master_bias_path)
    
    master_dark_path = os.path.join(data_folder_path, 'calibration/darks/master_dark.fit')
    mf.master_dark(dark_files, master_bias_path, master_dark_path)

    master_flats_save_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    os.makedirs(master_flats_save_folder, exist_ok=True) 
    flat_file_groups = {
        'visual': visual_flat_files,
        'blue': blue_flat_files,
        'red': red_flat_files
    }
    for filter_name, file_list in flat_file_groups.items():
        save_path = os.path.join(master_flats_save_folder, f'master_flat_{filter_name}.fit')
        mf.master_flat(file_list, master_bias_path, master_dark_path, save_path) 
    
    master_flats_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    filter_names = flat_file_groups.keys()

    mf.process_images_in_folder(science_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)
    mf.process_images_in_folder(standard_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)

    pad_val = 150 
    star_coords_main = [1180, 1400, 3700, 3800]
    bg_coords_main = [920, 1000, 3660, 3740]
    align_and_stack_folder(science_folder_path, star_coords_main, bg_coords_main, pad_val)
    
    star_coords_std = [2650, 2900, 2050, 2250]
    bg_coords_std = [2825, 2925, 1990, 2060]
    align_and_stack_folder(standard_folder_path, star_coords_std, bg_coords_std, pad_val)

    master_stack_paths = glob.glob(os.path.join(science_folder_path, "master_stack_*.fit"))
    ref_filter_name = 'blue' 
    master_ref_path = os.path.join(science_folder_path, f"master_stack_{ref_filter_name.lower()}.fit")


    star_coords_main = [1250, 1450, 3800, 4000]
    bg_coords_main = [1230, 1280, 3750, 3800]
    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []
    for stack_path in master_stack_paths:
        files_to_align.append(stack_path)
        if stack_path == master_ref_path:
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            shifts = mf.centroiding(stack_path, master_ref_path)
            master_shifts_x.append(shifts[0])
            master_shifts_y.append(shifts[1])
            
    for i, stack_path in enumerate(files_to_align):
        base_name = os.path.basename(stack_path)
        aligned_save_path = os.path.join(science_folder_path, f"aligned_{base_name}")
        mf.shifting([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val, aligned_save_path)
    return
