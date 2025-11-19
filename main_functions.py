import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
from scipy.ndimage import shift as scipy_shift
import scipy.signal
import os
import glob
import helper_functions as h

# these are all of the main function which will be used:

# this makes the master_bias frame
def master_bias(filelist, save_path):
    '''
    Creates the master bias frame.

    Paramters
    ---------
    filelist: List
        List of the bias frame fits files to create the master bias from
    save_path: String
        Path to where the master bias file will be saved

    Returns
    -------
    Array
        3D array containing the data of the created master bias
    '''
    #creates median stacked frame using helper functions
    master_bias =  h.mediancombine(filelist) 
    h.file_save(save_path, master_bias)
    return master_bias

def master_dark(filelist, master_bias_path, save_path):
    '''
    Makes the master dark frame. 
    NOTE: This does not sort files by filter, so only use on files with the same filter.

    Parameters
    ----------
    filelist: List
        List of the dark frame fits files to create the master dark from
    master_bias: Array
        3D array containing the data of the master bias frame
    save_path: String
        Path to where the master dark file will be saved
        
    Returns
    -------
    Array
        3D array containing the data of the created master dark frame
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
    Makes the master flat frame.

    Parameters
    ----------
    flat_files: List
        List of the flat frame fits files to create the master flat from
    master_bias: Array
        3D array containing the data of the master bias frame
    master_dark: Array
        3D array containing the data of the master dark frame
    save_path: String
        Path to where the master flat file will be saved
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

def image_processing(path_image, master_bias_path, master_dark_path, master_flats_folder, save_path, crop_val = 0): 
    '''
    Processes an image by correcting with the master bias, master dark, and master flat frames.
    Then masks (sets to np.nan) a certain number of pixels in the y, as counted from the top of a ds9 image (if needed)

    Parameters:

    '''
    header = fits.getheader(path_image)
    exptime = header["EXPTIME"]
    filter_image = header["FILTER"]
    
    bias_sub_science = h.bias_subtract(path_image, master_bias_path)
    d_b_subtracted = h.dark_subtract(bias_sub_science, master_dark_path, exptime)
    reduced_image = h.flat_correct(d_b_subtracted, master_flats_folder, filter_image)

    # Checks if removal exists and is greater than 0
    if crop_val > 0:
        c = int(crop_val)
        reduced_image = reduced_image[:-c, : ]
    h.file_save(save_path, reduced_image, header)
    return reduced_image
    
def centroiding(image_path_science, image_path_ref, star_coords, background_coords):
    '''
    Finds shifts needed to align an image with the given reference image using the centroiding method.

    Parameters:
    -----------
    image_path_science: String
        Path to calibrated science image
    image_path_ref: String
        Path to reference image
    star_coords: List
        List of minimum and maximum x and y values to use as dimensions for cutouts of a star in the images
    background_coords: List
        List of minimum and maximum x and y values to use as dimensions for cutouts of a background patch in the images

    Returns:
    --------
    List
        List of the x and y values needed to align image with the reference
    '''
    #uses box helper function to create cutouts
    new_box, new_background, new_xmin, new_ymin = h.box_maker(image_path_science, star_coords, background_coords)
    ref_box, ref_background, ref_xmin, ref_ymin = h.box_maker(image_path_ref, star_coords, background_coords)

    #displays cutouts, giving option to terminate function if created cutout is awful
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(ref_box)
    axes[0, 1].imshow(ref_background)
    axes[1, 0].imshow(new_box)
    axes[1, 1].imshow(new_background)
    plt.show()
    #try:
        #user_choice = input("Do they looked fucked (y/n): ")
        #if user_choice.lower().strip().startswith('y'):
            #return None
    #except Exception as e:
        #pass

    #finds cutoff values with sigma_finder helper function
    three_sigma_ref, avg_background_ref, std_bckground_ref = h.sigma_finder(ref_background)
    three_sigma_new, avg_background_new, std_bckground_new  = h.sigma_finder(new_background)
    #removes background pixel values with the star_isolator helper function
    new_star_box = h.star_isolator(new_box, three_sigma_new, avg_background_new)
    ref_star_box = h.star_isolator(ref_box, three_sigma_ref, avg_background_ref)
    #finds coordinates of the centroids in the input and reference image star cutouts
    x_coord_new, y_coord_new = h.star_finder(new_star_box, new_xmin, new_ymin)
    x_coord_ref, y_coord_ref = h.star_finder(ref_star_box, ref_xmin, ref_ymin)
    #calculates shift needed to align image with reference
    shift_x = x_coord_ref - x_coord_new
    shift_y = y_coord_ref - y_coord_new
    #creates list to hold shift values
    final_shifts = [shift_x, shift_y]
    print(final_shifts, "x then y")
    return final_shifts

def shifting(list_image_paths, x_shift, y_shift, pad_val, save_path):
    '''
    Shifts a list of images to align them, and then stacks them together.

    Parameters:
    ----------
    list_image_paths: List
        List containing the file paths for the images to align
    x_shift: List
        List of values to shift the x-coordinates of an image by
    y_shift: List
        List of values to shift the y-coordinates of an image by
    pad_val: Float
        Value of how much padding you wish to add to each image
    save_path: String
        Path to the desired save location

    Returns:
    --------
    Array
        Data array for the final aligned and stacked image
    '''
    #checks if you've got the right matching number of shifts or images
    if len(list_image_paths) != len(x_shift):
        print("Inputs are wrong womp womp")
        return

    # determines the final padded shape
    sample_data = fits.getdata(list_image_paths[0])
    padded_shape = np.pad(sample_data, pad_val, 'constant').shape

    #Pads and shifts images
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
    #creates median stacked image 
    final_median_image = np.nanmedian(stack, axis=2)
    h.file_save(save_path, final_median_image, fits.getheader(list_image_paths[0]))
    return final_median_image


def process_images_in_folder(base_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder):
    '''
    Processes all images within a folder.
    
    Parameters:
    ----------
    base_folder_path: String
        Path to folder with images you want to process
    filter_names: List
        List of filter names 
    master_bias_path: String
        Path to the master bias frame
    master_dark_path: String
        Path to the master dark frame
    master_flats_folder:
        Path to folder with master flat frames for each filter
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
            image_processing(image_path, master_bias_path, master_dark_path, master_flats_folder, final_save_path, 700)
    return

def cross_correlation_shifts(image_path_science, image_path_ref):
    '''
    Finds shifts needed to align and image with a reference using the cross correlation method.

    Parameters:
    -----------
    image_path_science: String
        Path to processes science image
    image_path_ref: String
        Path to reference image

    Returns:
    --------
    List
        List containing the x and y values needed to align image with the reference
    '''
    # Get image data
    im1 = fits.getdata(image_path_ref)
    im2 = fits.getdata(image_path_science) 

    min_rows = min(im1.shape[0], im2.shape[0])
    min_cols = min(im1.shape[1], im2.shape[1])

    # This helper calculates the start indices to crop exactly from the middle, will move it later if this works
    def crop_center(img, target_rows, target_cols):
        current_rows, current_cols = img.shape
        start_row = (current_rows - target_rows) // 2
        start_col = (current_cols - target_cols) // 2
        return img[start_row : start_row + target_rows, start_col : start_col + target_cols]

    # Apply the center crop
    im1 = crop_center(im1, min_rows, min_cols)
    im2 = crop_center(im2, min_rows, min_cols)
    
    valid_rows_1 = ~np.isnan(im1).any(axis=1)
    valid_rows_2 = ~np.isnan(im2).any(axis=1)
    valid_mask = valid_rows_1 & valid_rows_2
    im1_cut = im1[valid_mask]
    im2_cut = im2[valid_mask]
    im1_gray = im1_cut.astype('float')
    im2_gray = im2_cut.astype('float')
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same') 
    peak_corr_index = np.argmax(corr_image)
    corr_tuple = np.unravel_index(peak_corr_index, corr_image.shape)
    yshift = corr_tuple[0] - corr_image.shape[0]/2.
    xshift = corr_tuple[1] - corr_image.shape[1]/2.
    final_shifts = [xshift, yshift]
    print(final_shifts, "x then y") 
    return final_shifts
    
def shifting_fft(list_image_paths, x_shift, y_shift, pad_val, save_path):
    '''
    Shifts a list of images using a fast fourier transform, and then stacks them together.
    Parameters:
    -----------
    list_image_paths: List
        List of paths to images you want to shift
      x_shift: List
        List of values to shift the x-coordinates of an image by
    y_shift: List
        List of values to shift the y-coordinates of an image by
    pad_val: Float
        Value of how much padding you wish to add to each image
    save_path: String
        Path to the desired save location
        
    Returns:
    --------
    Array
        Data array of the final aligned and stacked image
    '''
    # checks if you've got the right matching number of shifts or images
    if len(list_image_paths) != len(x_shift):
        print("Inputs are wrong womp womp")
        return

    shifted_arrays = []

    # shifts each image in the input list
    for index, filename in enumerate(list_image_paths):
        im = fits.getdata(filename)
        shifted_im = np.roll(np.roll(im, int(y_shift[index]), axis=0), int(x_shift[index]), axis=1)
        shifted_arrays.append(shifted_im)
    final_median_image = h.mediancombine(shifted_arrays)
    max_x_shift = int(np.max(np.abs(x_shift)))
    max_y_shift = int(np.max(np.abs(y_shift)))
    
    if (max_x_shift > 0) & (max_y_shift > 0): 
        final_median_image = final_median_image[max_y_shift:-max_y_shift, max_x_shift:-max_x_shift]
    h.file_save(save_path, final_median_image, fits.getheader(list_image_paths[0]))
    return final_median_image

def shifting_masters(list_image_paths, x_shift, y_shift, ref_image_path, save_path):
    '''
    '''
    if len(list_image_paths) != len(x_shift):
        print("Inputs are wrong womp womp")
        return

    ref_data = fits.getdata(ref_image_path)
    ref_h, ref_w = ref_data.shape
    ref_header = fits.getheader(ref_image_path)

    for index, filename in enumerate(list_image_paths):
        target_data = fits.getdata(filename)
        t_h, t_w = target_data.shape
        if t_h < ref_h:
            # Target is SHORTER -> Pad Top/Bottom
            diff = ref_h - t_h
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            target_data = np.pad(target_data, ((pad_top, pad_bottom), (0, 0)), 
                                 mode='constant', constant_values=np.nan)
        elif t_h > ref_h:
            diff = t_h - ref_h
            start_crop = diff // 2
            target_data = target_data[start_crop : start_crop + ref_h, :]
        # Re-check shape
        curr_h, curr_w = target_data.shape
        if curr_w < ref_w:
            diff = ref_w - curr_w
            pad_left = diff // 2
            pad_right = diff - pad_left
            target_data = np.pad(target_data, ((0, 0), (pad_left, pad_right)), 
                                 mode='constant', constant_values=np.nan)
        elif curr_w > ref_w:
            # Target is WIDER -> Crop Left/Right
            diff = curr_w - ref_w
            start_crop = diff // 2
            target_data = target_data[:, start_crop : start_crop + ref_w]
        shift_y_int = int(y_shift[index])
        shift_x_int = int(x_shift[index])
        shifted_image = np.roll(target_data, shift_y_int, axis=0)
        shifted_image = np.roll(shifted_image, shift_x_int, axis=1)
        h.file_save(save_path, shifted_image, ref_header)
        return shifted_image
