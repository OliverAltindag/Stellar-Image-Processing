import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
import os
import glob
import helper_functions as h
import main_functions as mf

# Very specific helper fucntions

def sort_and_align_files(science_folder_path):
    '''
    Sorts science images by filter and gathers the list of shifts needed to align them.
    
    Parameters:
    -----------
    science_folder_path: String
        Path to the folder with science images to be sort and find shifts for
    
    Returns:
    --------
    List
        List of paths for images in each filter
    List
        List containing the shifts needed to align the images for each filter
    '''
    #collects calibrated files
    search_pattern = os.path.join(science_folder_path, "**", "fdb_*.fit")
    fdb_science_files = glob.glob(search_pattern, recursive=True)

    filter_files = {
        'Visual': [],
        'Blue': [],
        'Red': []
    }
    #sorts image paths by filter
    for path in fdb_science_files:
        filter_name = fits.getheader(path)["FILTER"].strip()
        if filter_name in filter_files:
            filter_files[filter_name].append(path)

    all_shifts = {}
    for filter_name, file_list in filter_files.items():
        current_shift_list = []
        # Uses the first file as the reference
        image_path_ref = file_list[0] 
        #gets shifts
        for image_path in file_list:
            x_y_shifts = mf.cross_correlation_shifts(image_path, image_path_ref)
            if x_y_shifts is None:
                print(f"User aborted alignment for filter: {filter_name}. Check star/background boxes.")
                current_shift_list = [] # Clear any shifts
                break
            current_shift_list.append(x_y_shifts)    
        all_shifts[filter_name] = current_shift_list
    return filter_files, all_shifts

def stack_all_filters(folder_path, filter_files, all_shifts, pad_val):
    '''
    Shifts and stacks images in each filter.
    
    Parameters:
    -----------
    folder_path: String
        Path to the folder with the files to be processed
    filter_files: List
        List of file paths for images in each filter
    all_shifts: List
        List containing the shifts needed to align the images for each filter
    pad_val: Float
        Value to pad the images by when shifting and stacking

    '''
    #shifts and stacks images in each filter
    for filter_name in filter_files.keys():
        file_list = filter_files.get(filter_name, [])
        shifts_list = all_shifts.get(filter_name, [])
        save_path = os.path.join(folder_path, f'master_stack_{filter_name.lower()}.fit')
        x_shifts = [s[0] for s in shifts_list]
        y_shifts = [s[1] for s in shifts_list]
        mf.shifting_fft(file_list, x_shifts, y_shifts, pad_val, save_path)
    return
    
def align_and_stack_folder(folder_path, pad_val):
    '''
    Shifts and stacks all images in a given folder.
    
    Parameters:
    -----------
    folder_path: String
        Path to the folder with the files to be processed
    pad_val: Float
        Value to pad the images by when shifting and stacking

    '''
    filter_files, all_shifts = sort_and_align_files(folder_path)
    stack_all_filters(folder_path, filter_files, all_shifts, pad_val)
    return

# The actual image reduction and stacking is done

def reduction(data_folder_path, science_images_folder):
    '''
    Reduces/processes all science images in a single function.

    Parameters:
    -----------
    data_folder_path: String
        Path to folder containing the calibration frames
    science_images_folder: String
        Path to the folder containing the science images to reduce
    '''
    standard_images_subfolder = "standard"
    standard_folder_path = os.path.join(data_folder_path, standard_images_subfolder)
    science_folder_path = os.path.join(data_folder_path, science_images_folder)

    #collects calibration frame files
    bias_files = glob.glob(os.path.join(data_folder_path, 'calibration/biasframes', '*.fit'))
    dark_files = glob.glob(os.path.join(data_folder_path, 'calibration/darks', '*.fit'))
    visual_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/visual', '*.fit'))
    blue_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/blue', '*.fit'))
    red_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/red', '*.fit'))
    
    #creates master calibration frames
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

    #calibrates images in each filter
    mf.process_images_in_folder(science_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)
    mf.process_images_in_folder(standard_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)

    #shifts and stacks images
    pad_val = 150 
    align_and_stack_folder(science_folder_path, pad_val)
    align_and_stack_folder(standard_folder_path, pad_val)

    master_stack_paths = glob.glob(os.path.join(science_folder_path, "master_stack_*.fit"))
    ref_filter_name = 'red' 
    master_ref_path = os.path.join(science_folder_path, f"master_stack_{ref_filter_name.lower()}.fit")

    #creates final image
    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []
    for stack_path in master_stack_paths:
        files_to_align.append(stack_path)
        if stack_path == master_ref_path:
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            shifts = mf.cross_correlation_shifts(stack_path, master_ref_path)
            master_shifts_x.append(shifts[0])
            master_shifts_y.append(shifts[1])
            
    for i, stack_path in enumerate(files_to_align):
        base_name = os.path.basename(stack_path)
        aligned_save_path = os.path.join(science_folder_path, f"aligned_{base_name}")
        mf.shifting_masters([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val, aligned_save_path)
    return
