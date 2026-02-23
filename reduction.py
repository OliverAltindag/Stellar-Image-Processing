import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
import os
import glob
import helper_functions as h
import main_functions as mf

# Very specific helper functions, not main functions
# here to keep the actual reduction() clean, and easy to debug
def sort_and_align_files(science_folder_path):
    '''
    Sorts science images by filter and gathers the list of shifts needed to align them.
    '''
    # collects calibrated files
    search_pattern = os.path.join(science_folder_path, "**", "fdb_*.fit")
    fdb_science_files = glob.glob(search_pattern, recursive=True)

    # gets the possible filter types, so files can be added in later
    filter_files = {'Visual': [], 'Blue': [], 'Red': []}
    
    # sorts image paths by filter, and adds them to the lists above
    for path in fdb_science_files:
        filter_name = fits.getheader(path)["FILTER"].strip()
        if filter_name in filter_files:
            filter_files[filter_name].append(path)

    all_shifts = {}
    for filter_name, file_list in filter_files.items():
        if not file_list: continue # skip empty filters
        current_shift_list = []
        # Uses the first file as the reference
        image_path_ref = file_list[0] 
        # gets shifts using main functions
        for image_path in file_list:
            x_y_shifts = mf.cross_correlation_shifts(image_path, image_path_ref)
            if x_y_shifts is None:
                print(f"User aborted alignment for filter: {filter_name}. shame :(")
                current_shift_list = [] # Clear any shifts
                break
            # will append the shifts list to keep track of historical ones
            current_shift_list.append(x_y_shifts) 
        # puts it into a sorted dictionary
        all_shifts[filter_name] = current_shift_list
    return filter_files, all_shifts

def stack_all_filters(folder_path, filter_files, all_shifts, pad_val):
    '''
    Shifts and stacks images in each filter.
    '''
    #shifts and stacks images in each filter
    for filter_name in filter_files.keys():
        # gets the filter and the corresponding shifts
        file_list = filter_files.get(filter_name, [])
        shifts_list = all_shifts.get(filter_name, [])
        if not file_list or not shifts_list: continue
        
        # makes the save path
        save_path = os.path.join(folder_path, f'master_stack_{filter_name.lower()}.fit') # critical this is lower case for code logic
        x_shifts = [s[0] for s in shifts_list]
        y_shifts = [s[1] for s in shifts_list]
        # performs the shifts using main function
        mf.shifting_fft(file_list, x_shifts, y_shifts, pad_val, save_path)
    return

def align_and_stack_folder(folder_path, pad_val):
    '''
    Shifts and stacks all images in a given folder.
    '''
    # calls the fucntions above to shift and stack the images
    filter_files, all_shifts = sort_and_align_files(folder_path)
    stack_all_filters(folder_path, filter_files, all_shifts, pad_val)
    return

def _centroid_alignment_pass(file_list, star_coords, bg_coords, pad_val):
    '''
    Internal helper to avoid repeating the centroiding loop 4 times.
    Logic remains identical to original code.
    '''
    if not file_list: return
    
    file_list.sort() # Ensure consistent reference
    master_ref_path = file_list[0]
    master_shifts_x = []
    master_shifts_y = []

    for stack_path in file_list:
        if stack_path == master_ref_path:
            # Reference image gets zero shift (SAME REFERENCE STAR FOR ALL)
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            # All other images aligned here, using the main functions
            shifts = mf.centroiding(stack_path, master_ref_path, star_coords, bg_coords)
            if shifts is not None:
                master_shifts_x.append(shifts[0])
                master_shifts_y.append(shifts[1])
            else:
                # insallah this never happens
                print(f"Centroiding failed for {stack_path}, using zero shift")
                master_shifts_x.append(0.0)
                master_shifts_y.append(0.0)

    # apply shifts to align all images using our main functions
    for i, stack_path in enumerate(file_list):
        mf.shifting_master_cen([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val)

# The actual image reduction and stacking is done in this function
def reduction(data_folder_path, science_images_folder):
    '''
    Reduces/processes all science images in a single function.
    '''
    base_path = data_folder_path
    science_folder_path = os.path.join(base_path, "target")
    standard_folder_path = os.path.join(base_path, "standard")
    calibration_folder_path = os.path.join(base_path, "calibration")
    
    print("Cleaning up old files recursively (preserving initial files)...")
    for root_folder in [science_folder_path, standard_folder_path, calibration_folder_path]:    
        if os.path.exists(root_folder):
            for current_dir, subdirs, files in os.walk(root_folder):
                for filename in files:
                    # for us this is jno but will chnage depending on your data labelling
                    if not filename.startswith(("jno", "flat")):
                        file_path = os.path.join(current_dir, filename)
                        try:
                            os.remove(file_path)
                        except OSError as e:
                            print(f"Error deleting {file_path}: {e}")
                            
    # now that its just raw data it can be processed/reprocessed    
    standard_images_subfolder = "standard"
    standard_folder_path = os.path.join(data_folder_path, standard_images_subfolder)
    science_folder_path = os.path.join(data_folder_path, science_images_folder)

    # gets the needed file data and the file list
    bias_files = glob.glob(os.path.join(data_folder_path, 'calibration/biasframes', 'jno*.fit'))
    dark_files = glob.glob(os.path.join(data_folder_path, 'calibration/darks', 'jno*.fit'))
    
    flat_file_groups = {
        'visual': glob.glob(os.path.join(data_folder_path, 'calibration/flats/visual', 'flat*.fit')),
        'blue': glob.glob(os.path.join(data_folder_path, 'calibration/flats/blue', 'flat*.fit')),
        'red': glob.glob(os.path.join(data_folder_path, 'calibration/flats/red', 'flat*.fit'))
    }

    # creates the master bias/dark
    master_bias_path = os.path.join(data_folder_path, 'calibration/biasframes/master_bias.fit')
    mf.master_bias(bias_files, master_bias_path)
    master_dark_path = os.path.join(data_folder_path, 'calibration/darks/master_dark.fit')
    mf.master_dark(dark_files, master_bias_path, master_dark_path)

    # creates the master flat for each filter in a NEW folder called masters
    master_flats_save_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    os.makedirs(master_flats_save_folder, exist_ok=True) 

    for filter_name, file_list in flat_file_groups.items():
        save_path = os.path.join(master_flats_save_folder, f'master_flat_{filter_name}.fit')
        mf.master_flat(file_list, master_bias_path, master_dark_path, save_path) 

    master_flats_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    filter_names = list(flat_file_groups.keys())

    # does image reduction on the standard star and the target
    mf.process_images_in_folder(science_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)
    mf.process_images_in_folder(standard_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)

    # created the master stack images in each filter
    pad_val = 150 
    align_and_stack_folder(science_folder_path, pad_val)
    align_and_stack_folder(standard_folder_path, pad_val)

    # Adaptive trimming block
    master_stack_paths = glob.glob(os.path.join(science_folder_path, "master_stack_*.fit"))
    min_height, min_width = float('inf'), float('inf')
    for stack_path in master_stack_paths:
        data = fits.getdata(stack_path)
        h_img, w_img = data.shape
        min_height, min_width = min(min_height, h_img), min(min_width, w_img)
    
    print(f"Adaptive trimming to: {min_width} x {min_height}") 

    trimmed_stack_paths = []
    for stack_path in master_stack_paths:
        data, header = fits.getdata(stack_path), fits.getheader(stack_path)
        curr_h, curr_w = data.shape
        start_y, start_x = (curr_h - min_height) // 2, (curr_w - min_width) // 2
        cropped_data = data[start_y:start_y + min_height, start_x:start_x + min_width]
        
        header['NAXIS1'], header['NAXIS2'] = cropped_data.shape[1], cropped_data.shape[0]
        trimmed_name = os.path.basename(stack_path).replace("master_stack_", "trimmed_master_stack_")
        trimmed_path = os.path.join(os.path.dirname(stack_path), trimmed_name)
        h.file_save(trimmed_path, cropped_data, header)
        trimmed_stack_paths.append(trimmed_path)

    # science Pass 1
    _centroid_alignment_pass(trimmed_stack_paths, [2755, 2880, 1260, 1310], [1920, 1970, 1700, 1750], 150)
    
    # science Pass 2 (Double alignment)
    science_pass2_files = glob.glob(os.path.join(science_folder_path, "aligned_trimmed_master_stack_*.fit"))
    _centroid_alignment_pass(science_pass2_files, [2171, 2201, 2100, 2130], [1400, 1450, 2325, 2375], 150)

    # standard Pass 1
    std_pass1_files = glob.glob(os.path.join(standard_folder_path, "master_stack_*.fit"))
    _centroid_alignment_pass(std_pass1_files, [2690, 2765, 1675, 1730], [1475, 1525, 1575, 1625], 150)

    # standard Pass 2 (double)
    std_pass2_files = glob.glob(os.path.join(standard_folder_path, "aligned_master_stack_*.fit"))
    _centroid_alignment_pass(std_pass2_files, [2405, 2455, 1500, 1550], [2240, 2290, 1550, 1605], 150)
    
    return
