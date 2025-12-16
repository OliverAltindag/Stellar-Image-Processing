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
    
    Parameters:
    -----------
    science_folder_path: String
        Path to the folder with science images to be sort and find shifts for
    
    Returns:
    --------
    filter_files: List
        List of paths for images in each filter
    all_shifts: List
        List containing the shifts needed to align the images for each filter
    '''
    # collects calibrated files
    search_pattern = os.path.join(science_folder_path, "**", "fdb_*.fit")
    fdb_science_files = glob.glob(search_pattern, recursive=True)

    # gets the possible filter types, so files can be added in later
    filter_files = {
        'Visual': [],
        'Blue': [],
        'Red': []
    }
    # sorts image paths by filter, and adds them to the lists above
    for path in fdb_science_files:
        filter_name = fits.getheader(path)["FILTER"].strip()
        if filter_name in filter_files:
            filter_files[filter_name].append(path)

    all_shifts = {}
    for filter_name, file_list in filter_files.items():
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
        
    Returns:
    -----------
    None

    '''
    #shifts and stacks images in each filter
    for filter_name in filter_files.keys():
        # gets the filter and the corresponding shifts
        file_list = filter_files.get(filter_name, [])
        shifts_list = all_shifts.get(filter_name, [])
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
    
    Parameters:
    -----------
    folder_path: String
        Path to the folder with the files to be processed
    pad_val: Float
        Value to pad the images by when shifting and stacking
        
    Returns:
    -----------
    None

    '''
    # calls the fucntions above to shift and stack the images
    filter_files, all_shifts = sort_and_align_files(folder_path)
    stack_all_filters(folder_path, filter_files, all_shifts, pad_val)
    return



# The actual image reduction and stacking is done in this function
# for all of the scientific data
# this auto deletes all old filtered files to make re running easy
def reduction(data_folder_path, science_images_folder):
    '''
    Reduces/processes all science images in a single function.

    Parameters:
    -----------
    data_folder_path: String
        Path to folder containing the calibration frames
    science_images_folder: String
        Path to the folder containing the science images to reduce

    Returns:
    -----------
    None
    '''
    # gets the folders in the architecture so files can be deleted
    base_path = data_folder_path
    # did it like this for convenience but it is already defined in the read in line so 
    # no difference
    science_folder_path = os.path.join(base_path, "target")
    standard_folder_path = os.path.join(base_path, "standard")
    calibration_folder_path = os.path.join(base_path, "calibration")
    
    print("Cleaning up old files recursively (preserving initial files)...")
    # iterate through the main categories
    for root_folder in [science_folder_path, standard_folder_path, calibration_folder_path]:    
        # Check if the main folder exists first
        if os.path.exists(root_folder):
            # os.walk() recursively goes into subfolders
            # or it should at least
            # worked when I did it
            for current_dir, subdirs, files in os.walk(root_folder):
                for filename in files:
                    # for us this is jno but will chnage depending on your data labelling
                    # this assumes it was all imaging was labeled with the same tag
                    if not filename.startswith("jno"):
                        file_path = os.path.join(current_dir, filename)
                        try:
                            # for cleanliness didnt show the files being deleted but add
                            # if your heart desires
                            os.remove(file_path)
                        except OSError as e:
                            print(f"Error deleting {file_path}: {e}")
                            
    # now that its just raw data it can be processed/reprocessed    
    # using file architecture, we can merely find the standard star folder
    standard_images_subfolder = "standard"
    standard_folder_path = os.path.join(data_folder_path, standard_images_subfolder)
    science_folder_path = os.path.join(data_folder_path, science_images_folder)
    # gets the needed file data and the file list
    # could use filelist_creator, but wanted the added .fit portion
    bias_files = glob.glob(os.path.join(data_folder_path, 'calibration/biasframes', 'jno*.fit'))
    dark_files = glob.glob(os.path.join(data_folder_path, 'calibration/darks', 'jno*.fit'))
    visual_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/visual', 'jno*.fit'))
    blue_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/blue', 'jno*.fit'))
    red_flat_files = glob.glob(os.path.join(data_folder_path, 'calibration/flats/red', 'jno*.fit'))
    # creates the master bias
    master_bias_path = os.path.join(data_folder_path, 'calibration/biasframes/master_bias.fit')
    mf.master_bias(bias_files, master_bias_path)
    # creates the master dark
    master_dark_path = os.path.join(data_folder_path, 'calibration/darks/master_dark.fit')
    mf.master_dark(dark_files, master_bias_path, master_dark_path)
    # creates the master flat for each filter in a NEW folder called masters
    master_flats_save_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    os.makedirs(master_flats_save_folder, exist_ok=True) 
    flat_file_groups = {
        'visual': visual_flat_files,
        'blue': blue_flat_files,
        'red': red_flat_files
    }
    for filter_name, file_list in flat_file_groups.items():
        save_path = os.path.join(master_flats_save_folder, f'master_flat_{filter_name}.fit')
        # actually calls the fucntion here
        mf.master_flat(file_list, master_bias_path, master_dark_path, save_path) 
    # sets the path to the master flats folder to be used later and gets their names
    master_flats_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    filter_names = flat_file_groups.keys()
    # does image reduction on the standard star and the target
    mf.process_images_in_folder(science_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)
    mf.process_images_in_folder(standard_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)
    # created the master stack images in each filter
    pad_val = 150 
    align_and_stack_folder(science_folder_path, pad_val)
    align_and_stack_folder(standard_folder_path, pad_val)
    # gets the path to the master stacked images we just created
    master_stack_paths = glob.glob(os.path.join(science_folder_path, "master_stack_*.fit"))
    # trims the images, so they are all the same size again
    # the fft correlating process trimmed them all differently so want them identical again
    # this will help when padding and create better alignment
    # a critical step for RGB imaging
    min_height = float('inf')
    min_width = float('inf')
    for stack_path in master_stack_paths:
        data = fits.getdata(stack_path)
        height, width = data.shape
        min_height = min(min_height, height)
        min_width = min(min_width, width)
    print(f"Adaptive trimming to: {min_width} x {min_height}") # bc it wont work if not and ive done a ton of testing
    # trimmed versions, this now makes the alignment step more accurate with the padding
    trimmed_stack_paths = []
    for stack_path in master_stack_paths:
        # gets the data and the header
        data = fits.getdata(stack_path)
        header = fits.getheader(stack_path)
        # trims the image, keeps the central values to keep image integrity
        current_height, current_width = data.shape
        start_y = (current_height - min_height) // 2
        end_y = start_y + min_height
        start_x = (current_width - min_width) // 2
        end_x = start_x + min_width
        cropped_data = data[start_y:end_y, start_x:end_x]
        header['NAXIS1'] = cropped_data.shape[1]  # Update width
        header['NAXIS2'] = cropped_data.shape[0]  # Update height
        base_name = os.path.basename(stack_path)
        trimmed_name = base_name.replace("master_stack_", "trimmed_master_stack_")
        trimmed_path = os.path.join(os.path.dirname(stack_path), trimmed_name)
        # saves the data to be used in the future
        h.file_save(trimmed_path, cropped_data, header)
        trimmed_stack_paths.append(trimmed_path)
    # use the timrmed ones
    trimmed_stack_paths.sort()  # Sort to ensure consistent reference
    master_ref_path = trimmed_stack_paths[0]  
    pad_val = 150
    # notice this is 100 pixels wide and relatively centered
    star_coords_main = [2755, 2880, 1260, 1310]  # FINALLY WORKED DO NOT CHNAGE THESE PLEASE
    bg_coords_main = [1920, 1970, 1700, 1750] # finding these coordinates was the greatest triumph in this century 
    # initializes the empty list for the shifts
    # this will use centroiding
    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []
    for stack_path in trimmed_stack_paths:  
        files_to_align.append(stack_path)
        if stack_path == master_ref_path:
            # Reference image gets zero shift (SAME REFERENCE STAR FOR ALL), was an issue of this failing
            # hence has been fixed, no worries
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            # All other images aligned here, using the main functions
            shifts = mf.centroiding(stack_path, master_ref_path, star_coords_main, bg_coords_main)
            if shifts is not None:
                master_shifts_x.append(shifts[0])
                master_shifts_y.append(shifts[1])
            else:
                # insallah this never happens
                # our data succesfully went through this so is a file architecture issue if not
                print(f"Centroiding failed for {stack_path}, using zero shift")
                master_shifts_x.append(0.0)
                master_shifts_y.append(0.0)
    # apply shifts to align all images using our main functions
    # this one merely shifts, no median stacking 
    # sooooo important
    for i, stack_path in enumerate(files_to_align):
        mf.shifting_master_cen([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val)

    # aligns the standard star images
    # basically does everything above again witout trimming
    master_stack_paths = glob.glob(os.path.join(science_folder_path, "aligned_trimmed_master_stack_*.fit"))
    master_stack_paths.sort()
    master_ref_path = master_stack_paths[0]
    pad_val = 150
    star_coords_main = [2171, 2201, 2100, 2130]  # doing it twice bc they were so far apart before was still slightly off
    bg_coords_main = [1400, 1450, 2325, 2375]  
    # initializes the empty list for the shifts
    # this will use centroiding
    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []
    for stack_path in master_stack_paths:  
        files_to_align.append(stack_path)
        if stack_path == master_ref_path:
            # Reference image gets zero shift (SAME REFERENCE STAR FOR ALL), was an issue of this failing
            # hence has been fixed, no worries
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            # All other images aligned here, using the main functions
            shifts = mf.centroiding(stack_path, master_ref_path, star_coords_main, bg_coords_main)
            if shifts is not None:
                master_shifts_x.append(shifts[0])
                master_shifts_y.append(shifts[1])
            else:
                # insallah this never happens
                # our data succesfully went through this so is a file architecture issue if not
                print(f"Centroiding failed for {stack_path}, using zero shift")
                master_shifts_x.append(0.0)
                master_shifts_y.append(0.0)
    # apply shifts to align all images using our main functions
    # this one merely shifts, no median stacking 
    # sooooo important
    for i, stack_path in enumerate(files_to_align):
        mf.shifting_master_cen([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val)

    # aligns the standard star images
    # basically does everything above again witout trimming
    master_stack_paths = glob.glob(os.path.join(standard_folder_path, "master_stack_*.fit"))
    master_stack_paths.sort()
    master_ref_path = master_stack_paths[0]
    pad_val = 150
    star_coords_main = [2690, 2765, 1675, 1730]  
    bg_coords_main = [1475, 1525, 1575, 1625]  
    # initializes the empty list for the shifts
    # this will use centroiding
    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []
    for stack_path in master_stack_paths:  
        files_to_align.append(stack_path)
        if stack_path == master_ref_path:
            # Reference image gets zero shift (SAME REFERENCE STAR FOR ALL), was an issue of this failing
            # hence has been fixed, no worries
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            # All other images aligned here, using the main functions
            shifts = mf.centroiding(stack_path, master_ref_path, star_coords_main, bg_coords_main)
            if shifts is not None:
                master_shifts_x.append(shifts[0])
                master_shifts_y.append(shifts[1])
            else:
                # insallah this never happens
                # our data succesfully went through this so is a file architecture issue if not
                print(f"Centroiding failed for {stack_path}, using zero shift")
                master_shifts_x.append(0.0)
                master_shifts_y.append(0.0)
    # apply shifts to align all images using our main functions
    # this one merely shifts, no median stacking 
    # sooooo important
    for i, stack_path in enumerate(files_to_align):
        mf.shifting_master_cen([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val)

    # aligns the standard star images
    # basically does everything above again witout trimming
    master_stack_paths = glob.glob(os.path.join(standard_folder_path, "aligned_master_stack_*.fit"))
    master_stack_paths.sort()
    master_ref_path = master_stack_paths[0]
    pad_val = 150
    star_coords_main = [2405, 2455, 1500, 1550]  # just another one in the center ish
    bg_coords_main = [2240, 2290, 1550, 1605]  
    # initializes the empty list for the shifts
    # this will use centroiding
    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []
    for stack_path in master_stack_paths:  
        files_to_align.append(stack_path)
        if stack_path == master_ref_path:
            # Reference image gets zero shift (SAME REFERENCE STAR FOR ALL), was an issue of this failing
            # hence has been fixed, no worries
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            # All other images aligned here, using the main functions
            shifts = mf.centroiding(stack_path, master_ref_path, star_coords_main, bg_coords_main)
            if shifts is not None:
                master_shifts_x.append(shifts[0])
                master_shifts_y.append(shifts[1])
            else:
                # insallah this never happens
                # our data succesfully went through this so is a file architecture issue if not
                print(f"Centroiding failed for {stack_path}, using zero shift")
                master_shifts_x.append(0.0)
                master_shifts_y.append(0.0)
    # apply shifts to align all images using our main functions
    # this one merely shifts, no median stacking 
    # sooooo important
    for i, stack_path in enumerate(files_to_align):
        mf.shifting_master_cen([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val)
    
    return
