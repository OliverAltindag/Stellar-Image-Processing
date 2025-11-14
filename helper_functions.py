import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import shift as scipy_shift
import main_functions as mf

# Helper Function Definitons will all be defined here:

def file_save(save_path, data, header = None):
    '''
    '''
    fits.writeto(save_path, data, header, overwrite = True)
    return

def filelist_creator(base_path, subfolder_path):
    '''
    '''
    file_path = os.path.join(base_path, subfolder_path, '*')
    file_list = glob.glob(file_path)
    return file_list
    
def mediancombine(filelist):
    '''
    '''
    if isinstance(filelist[0], str):
        # Number of files within the input list
        n = len(filelist)
        # gets the data for the first file in the list (intermediary)
        first_frame_data = fits.getdata(filelist[0])
        # get the shape of the data array in pixels
        imsize_y, imsize_x = first_frame_data.shape
        # create zero array of the same shape as above
        fits_stack = np.zeros((imsize_y, imsize_x , n))
        # fills the 3d array with the data form the first file
        for ii in range(0, n):
            im = fits.getdata(filelist[ii])
            fits_stack[:,:,ii] = im
    elif isinstance(filelist[0], np.ndarray):
        # Number of files within the input list
        n = len(filelist)
        # gets the data for the first file in the list (intermediary)
        first_frame_data = filelist[0]
        # get the shape of the data array in pixels
        imsize_y, imsize_x = first_frame_data.shape
        # create zero array of the same shape as above
        fits_stack = np.zeros((imsize_y, imsize_x , n))
        # fills the 3d array with the data form the first file
        for ii in range(0, n):
            im = filelist[ii]
            fits_stack[:,:,ii] = im
    # takes the median array for each, creating the median array
    med_frame = np.median(fits_stack, axis = 2)
    return med_frame #returns the median stacked image

def bias_subtract(filename, path_to_bias): # better to use filename here than data directly (faster)
    '''
    '''
    data = fits.getdata(filename)
    # gets master bias data
    master = fits.getdata(path_to_bias)
    # subtracts master bias from input data
    bias_subtracted = data - master
    return bias_subtracted

def dark_subtract(data, master_dark_path, scale_multiple):
    '''
    '''
    master_dark = fits.getdata(master_dark_path)
    dark_subtracted = data - (master_dark * scale_multiple)
    return dark_subtracted

def normalization(data, file_path):
    '''
    '''
    header = fits.getheader(file_path)
    exptime = header["EXPTIME"]
    normalized = data / exptime
    return normalized

def flat_correct(science_data, master_flats_folder, filter_image):
    '''
    '''
    flat_filename = f"master_flat_{filter_image.strip()}.fits"
    master_flat_path = os.path.join(master_flats_folder, flat_filename)
    # Get the master flat data
    master_flat_data = fits.getdata(master_flat_path)
    master_flat_data[master_flat_data <= 0] = np.nan # if not there this did not work properly and would be an error :(
    # Divide the science image by the master flat
    flat_corrected = science_data / master_flat_data
    return flat_corrected

def box_maker(image_path, star_coords, background_coords):
    '''
    '''
    # gets the data
    original_data = fits.getdata(image_path)
    image_data = original_data.copy()
    star_xmin, star_xmax, star_ymin, star_ymax = star_coords[0], star_coords[1], star_coords[2], star_coords[3]
    bg_xmin, bg_xmax, bg_ymin, bg_ymax = background_coords[0], background_coords[1], background_coords[2], background_coords[3]
    # makes the boxes
    star_box = image_data[int(star_ymin):int(star_ymax), int(star_xmin):int(star_xmax)]
    background_box = image_data[int(bg_ymin):int(bg_ymax), int(bg_xmin):int(bg_xmax)]
    return star_box, background_box, star_xmin, star_ymin

def sigma_finder(box):
    '''
    '''
    # gets the cutoff value
    avg_background = np.mean(box)
    std_background = np.std(box)
    three_sigma_cutoff = 3 * std_background + avg_background
    return three_sigma_cutoff, avg_background, std_background

def star_isolator(box, cutoff, avg_background):
    '''
    '''
    star = box <= cutoff
    box -= avg_background
    box[star] = np.nan
    return box

def star_finder(box, xmin, ymin):
    '''
    '''
    num_rows, num_cols = box.shape
    y_indices, x_indices = np.indices((num_rows, num_cols))
    sum_I = np.nansum(box)
    x_sum_num = np.nansum(x_indices * box)
    y_sum_num = np.nansum(y_indices * box)
    x_coord = xmin + (x_sum_num / sum_I)
    y_coord = ymin + (y_sum_num / sum_I)
    return x_coord, y_coord

def sort_and_align_files(science_folder_path, star_coords, background_coords):
    '''
    '''
    search_pattern = os.path.join(science_folder_path, "**", "fdb_*.fits")
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
            x_y_shifts = centroiding(image_path, image_path_ref, star_coords, background_coords)
            current_shift_list.append(x_y_shifts)    
        all_shifts[filter_name] = current_shift_list
    return filter_files, all_shifts

def stack_all_filters(folder_path, filter_files, all_shifts, pad_val):
    '''
    '''
    for filter_name in filter_files.keys():
        file_list = filter_files.get(filter_name, [])
        shifts_list = all_shifts.get(filter_name, [])
        save_path = os.path.join(folder_path, f'master_stack_{filter_name.lower()}.fits')
        x_shifts = [s[0] for s in shifts_list]
        y_shifts = [s[1] for s in shifts_list]
        shifting(file_list, x_shifts, y_shifts, pad_val, save_path)
    return

def process_images_in_folder(base_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder):
    '''
    '''
    for filter_name in filter_names:
        filter_subfolder_path = os.path.join(base_folder_path, filter_name)
        # Find all .fits files in the subfolder
        raw_files = glob.glob(os.path.join(filter_subfolder_path, '*.fits'))
        for image_path in raw_files:
            base_filename = os.path.basename(image_path)
            # Skip files that have already been processed
            if "fdb_" in base_filename:
                continue
            new_filename = "fdb_" + base_filename
            final_save_path = os.path.join(filter_subfolder_path, new_filename)
            mf.image_processing(image_path, master_bias_path, master_dark_path, master_flats_folder, final_save_path)
