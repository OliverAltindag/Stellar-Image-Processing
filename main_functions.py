import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
import os
import glob
import helper_functions as h

# these are all of the main function which will be used:

# this makes the master_bias frame
def master_bias(filelist, save_path):
    """
    """
    #creates median stacked frame using helper functions
    master_bias =  h.mediancombine(filelist) 
    h.file_save(save_path, master_bias)
    return master_bias

# this one makes the master dark BUT need to run it for each exptime individually at the moment
def master_dark(filelist, master_bias_path, save_path):
    '''
    '''
    normalized_darks = []
    for file_path in filelist:
        bias_subtracted_image = h.bias_subtract(file_path, master_bias_path)
        normalized_sub_image = h.normalization(bias_subtracted_image, file_path)
        normalized_darks.append(normalized_sub_image)
    master_dark = h.mediancombine(normalized_darks)
    h.file_save(save_path, master_dark) 
    return master_dark

def master_flat(flat_files, master_bias_path, master_dark_path, save_path):
    '''
    '''
    dark_subtracted_normalized = []
    for flat_file_path in flat_files:
        flat_bias_sub = h.bias_subtract(flat_file_path, master_bias_path)
        normalized_flat = h.normalization(flat_bias_sub, flat_file_path)
        norm_flat_dark_subtracted = h.dark_subtract(normalized_flat, master_dark_path, 1)
        dark_subtracted_normalized.append(norm_flat_dark_subtracted)
    median_combine_image = h.mediancombine(dark_subtracted_normalized)
    master_flat = median_combine_image / np.median(median_combine_image)
    h.file_save(save_path, master_flat) 
    return master_flat

def image_processing(path_image, master_bias_path, master_dark_path, master_flats_folder, save_path): 
    '''
    '''
    header = fits.getheader(path_image)
    exptime = header["EXPTIME"]
    filter_image = header["FILTER"]
    bias_sub_science = h.bias_subtract(path_image, master_bias_path)
    d_b_subtracted = h.dark_subtract(bias_sub_science, master_dark_path, exptime)
    reduced_image = h.flat_correct(d_b_subtracted, master_flats_folder, filter_image)
    h.file_save(save_path, reduced_image, header)
    return

def centroiding(image_path_science, image_path_ref, star_coords, background_coords):
    '''
    '''
    new_box, new_background, new_xmin, new_ymin = h.box_maker(image_path_science, star_coords, background_coords)
    ref_box, ref_background, ref_xmin, ref_ymin = h.box_maker(image_path_ref, star_coords, background_coords)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(ref_box)
    axes[0, 1].imshow(ref_background)
    axes[1, 0].imshow(new_box)
    axes[1, 1].imshow(new_background)
    plt.show()
    try:
        user_choice = input("Do they looked fucked (y/n): ")
        if user_choice.lower().strip().startswith('y'):
            return None
    except Exception as e:
        pass

    three_sigma_ref, avg_background_ref, std_bckground_ref = h.sigma_finder(ref_background)
    three_sigma_new, avg_background_new, std_bckground_new  = h.sigma_finder(new_background)
    new_star_box = h.star_isolator(new_box, three_sigma_new, avg_background_new)
    ref_star_box = h.star_isolator(ref_box, three_sigma_ref, avg_background_ref)
    x_coord_new, y_coord_new = h.star_finder(new_star_box, new_xmin, new_ymin)
    x_coord_ref, y_coord_ref = h.star_finder(ref_star_box, ref_xmin, ref_ymin)
    shift_x = x_coord_ref - x_coord_new
    shift_y = y_coord_ref - y_coord_new
    final_shifts = [shift_x, shift_y]
    print(final_shifts, "x then y")
    return final_shifts

def shifting(list_image_paths, x_shift, y_shift, pad_val, save_path):
    '''
    '''
    if len(list_image_paths) != len(x_shift):
        print("Inputs are wrong womp womp")
        return

    # determines the final padded shape
    sample_data = fits.getdata(list_image_paths[0])
    padded_shape = np.pad(sample_data, pad_val, 'constant').shape
    
    num_images = len(list_image_paths)
    stack = np.zeros((padded_shape[0], padded_shape[1], num_images), dtype=np.float32)
    for i in range(num_images):
        current_image_path = list_image_paths[i]
        image_data = fits.getdata(current_image_path)
        image_data[np.isinf(image_data)] = 0.0
        image_data[np.isnan(image_data)] = 0.0
        padded_image = np.pad(image_data, pad_val, 'constant', constant_values = -1)
        shifted_padded_image = scipy_shift(padded_image, (y_shift[i], x_shift[i]), cval=-1)
        shifted_padded_image[shifted_padded_image <= -0.99] = np.nan
        stack[:,:,i] = shifted_padded_image
    final_median_image = np.nanmedian(stack, axis=2)
    h.file_save(save_path, final_median_image, fits.getheader(list_image_paths[0]))
    return final_median_image


def process_images_in_folder(base_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder):
    '''
    '''
    for filter_name in filter_names:
        filter_subfolder_path = os.path.join(base_folder_path, filter_name)
        # Find all .fits files in the subfolder
        raw_files = glob.glob(os.path.join(filter_subfolder_path, '*.fit'))
        for image_path in raw_files:
            base_filename = os.path.basename(image_path)
            # Skip files that have already been processed
            if "fdb_" in base_filename:
                continue
            new_filename = "fdb_" + base_filename
            final_save_path = os.path.join(filter_subfolder_path, new_filename)
            image_processing(image_path, master_bias_path, master_dark_path, master_flats_folder, final_save_path)
    return

def cross_correlation_shifts(image_path_science, image_path_ref):
    '''
    '''

    im1 = fits.getdata(image_path_ref)
    im2 = fits.getdata(image_path_science) # Mapped from original science image input
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)
    # Calculate the cross-correlation image
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same') 
    peak_corr_index = np.argmax(corr_image)
    corr_tuple = np.unravel_index(peak_corr_index, corr_image.shape)
    xshift = corr_tuple[0] - corr_image.shape[0]/2.
    yshift = corr_tuple[1] - corr_image.shape[1]/2.
    final_shifts = [xshift, yshift]
    print(final_shifts, "x then y") 
    return final_shifts

def shifting_fft(list_image_paths, x_shift, y_shift, pad_val, save_path):
    '''
    '''
    if len(list_image_paths) != len(x_shift):
        print("Inputs are wrong womp womp")
        return
    shifted_arrays = []
    for index, filename in enumerate(list_image_paths):
        im = fits.getdata(filename)
        shifted_im = np.roll(np.roll(im, int(y_shift[index]), axis=1), int(x_shift[index]), axis=0)
        shifted_arrays.append(shifted_im)
    final_median_image = mediancombine(shifted_arrays)
    max_x_shift = int(np.max(np.abs(x_shift)))
    max_y_shift = int(np.max(np.abs(y_shift)))
    if (max_x_shift > 0) & (max_y_shift > 0): 
        final_median_image = final_median_image[max_x_shift:-max_x_shift, max_y_shift:-max_y_shift]
    h.file_save(save_path, final_median_image, fits.getheader(list_image_paths[0]))
    return final_median_image

