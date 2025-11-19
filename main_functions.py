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
    # calls our helper function without defining a header
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
    # initializes a list for the normalized dark frames to be later processed
    normalized_darks = []
    for file_path in filelist:
        # bias subtract
        bias_subtracted_image = h.bias_subtract(file_path, master_bias_path)
        # normalizes
        normalized_sub_image = h.normalization(bias_subtracted_image, file_path)
        normalized_darks.append(normalized_sub_image)
    # takes the median combine value
    master_dark = h.mediancombine(normalized_darks)
    # saves the file without a header
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
    # initializes a list for the bias and dark subtracted frames
    dark_subtracted_normalized = []
    for flat_file_path in flat_files:
        # bias subtract
        flat_bias_sub = h.bias_subtract(flat_file_path, master_bias_path)
        # normalizes
        normalized_flat = h.normalization(flat_bias_sub, flat_file_path)
        # dark subtract with 1 for the scalar
        norm_flat_dark_subtracted = h.dark_subtract(normalized_flat, master_dark_path, 1)
        dark_subtracted_normalized.append(norm_flat_dark_subtracted)
    # takes the median combine of the newly made list
    median_combine_image = h.mediancombine(dark_subtracted_normalized)
    master_flat = median_combine_image / np.median(median_combine_image)
    # saves the file without a header
    # the frame has no exptime and adding no information is better for later information 
    # than wrong information
    h.file_save(save_path, master_flat) 
    return master_flat

def image_processing(path_image, master_bias_path, master_dark_path, master_flats_folder, save_path, crop_val = 0): 
    '''
    Processes an image by correcting with the master bias, master dark, and master flat frames.
    Then masks (sets to np.nan) a certain number of pixels in the y, as counted from the top of a ds9 image (if needed)

    Parameters:
    ----------
    path_image: String
        The path to the image we are processing
    master_bias_path: String
        The path to the master bias frame
    master_dark_path: String
        The path to the master dark frame
    master_flats_folder: String
        Path to where the master flat files will be saved
    save_path: String
        The path where the data should be saved later on
    crop_val: Integer
        A value to cut the top of the image off due to an unwanted artifact

    Returns:
    ----------
    reduced_image: Array
        The array of the reduced image data, which has been saved.
    '''
    # gets the filter for the image
    header = fits.getheader(path_image)
    exptime = header["EXPTIME"]
    filter_image = header["FILTER"]
    # performs the processsing, using helpers
    bias_sub_science = h.bias_subtract(path_image, master_bias_path)
    d_b_subtracted = h.dark_subtract(bias_sub_science, master_dark_path, exptime)
    reduced_image = h.flat_correct(d_b_subtracted, master_flats_folder, filter_image)
    # Checks if removal exists and is greater than 0
    if crop_val > 0:
        c = int(crop_val)
        reduced_image = reduced_image[:-c, : ]
    # saves the files, using helpers
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
    # uses box helper function to create cutouts
    new_box, new_background, new_xmin, new_ymin = h.box_maker(image_path_science, star_coords, background_coords)
    ref_box, ref_background, ref_xmin, ref_ymin = h.box_maker(image_path_ref, star_coords, background_coords)

    #displays cutouts, giving option to terminate function if created cutout is awful
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(ref_box)
    axes[0, 1].imshow(ref_background)
    axes[1, 0].imshow(new_box)
    axes[1, 1].imshow(new_background)
    plt.show()
    # because of the image quantity and cherry-picked coordinates, this was removed for efficiency
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

def process_images_in_folder(base_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder):
    '''
    Processes all images within a folder. This will be done per science object.
    
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
        
    Returns:
    ----------
        None  
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
    final_shifts: List
        List containing the x and y values needed to align image with the reference
    '''
    # Get image data
    im1 = fits.getdata(image_path_ref)
    im2 = fits.getdata(image_path_science) 
    # obtains the minimum shape amongst them
    min_rows = min(im1.shape[0], im2.shape[0])
    min_cols = min(im1.shape[1], im2.shape[1])
    # Apply the center crop
    im1 = h.crop_center(im1, min_rows, min_cols)
    im2 = h.crop_center(im2, min_rows, min_cols)
    # finds the bad rows and removes it from both images
    valid_rows_1 = ~np.isnan(im1).any(axis=1)
    valid_rows_2 = ~np.isnan(im2).any(axis=1)
    valid_mask = valid_rows_1 & valid_rows_2
    im1_cut = im1[valid_mask]
    im2_cut = im2[valid_mask]
    # removes the background from the remaining rows
    im1_gray = im1_cut.astype('float')
    im2_gray = im2_cut.astype('float')
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)
    # calculates the cross correlation bewteen the two images
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same') 
    # the best correlation is defined below
    peak_corr_index = np.argmax(corr_image)
    corr_tuple = np.unravel_index(peak_corr_index, corr_image.shape)
    # the shifts to make
    # NOTE: lab06 had a double negative error, which made the final answer correct but the code 
    # wrong. The logic was just minorly refined here
    yshift = corr_tuple[0] - corr_image.shape[0]/2.
    xshift = corr_tuple[1] - corr_image.shape[1]/2.
    final_shifts = [xshift, yshift]
    print(final_shifts, "x then y") 
    return final_shifts
    
def shifting_fft(list_image_paths, x_shift, y_shift, pad_val, save_path):
    '''
    Shifts a list of images using a fast fourier transform, and then stacks them together.
    This code also median combines them at the end.
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
    final_median_image: Array
        Data array of the final aligned and stacked image. It is also trimmed.
    '''
    # checks if you've got the right matching number of shifts or images
    if len(list_image_paths) != len(x_shift):
        print("Inputs are wrong womp womp")
        return
        
    # initializes an array of the shifted arrays
    shifted_arrays = []
    # shifts each image in the input list
    for index, filename in enumerate(list_image_paths):
        im = fits.getdata(filename)
        # uses np.roll so only whole number values can be entered
        shifted_im = np.roll(np.roll(im, int(y_shift[index]), axis=0), int(x_shift[index]), axis=1)
        shifted_arrays.append(shifted_im)
    # median combines them to make the master image for each filter
    final_median_image = h.mediancombine(shifted_arrays)
    # finds the maximum shifts performed to trim the image
    max_x_shift = int(np.max(np.abs(x_shift)))
    max_y_shift = int(np.max(np.abs(y_shift)))
    # here it saves the trimmed image
    if (max_x_shift > 0) & (max_y_shift > 0): 
        final_median_image = final_median_image[max_y_shift:-max_y_shift, max_x_shift:-max_x_shift]
    h.file_save(save_path, final_median_image, fits.getheader(list_image_paths[0]))
    return final_median_image

def shifting_master_cen(list_image_paths, x_shift, y_shift, pad_val, save_path):
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
    if len(list_image_paths) != len(x_shift):
        print("Inputs are wrong womp womp")
        return

    sample_data = fits.getdata(list_image_paths[0])
    # Pads and shifts images
    num_images = len(list_image_paths)
    for i in range(num_images):
        current_image_path = list_image_paths[i]
        image_data = fits.getdata(current_image_path)
        base_name = os.path.basename(current_image_path)
        unique_save_path = os.path.join(os.path.dirname(current_image_path), f"aligned_{base_name}") 
        padded_image = np.pad(image_data, pad_val, 'constant', constant_values = -1) # ask about this bc if i do nan with no median combine it just return all nan
        # scipy_shift expects (Y_shift, X_shift)
        shifted_padded_image = scipy_shift(padded_image, (y_shift[i], x_shift[i]), cval=-1)
        # shifted_padded_image[shifted_padded_image <= -0.99] = np.nan
        h.file_save(unique_save_path, shifted_padded_image, fits.getheader(current_image_path))
    return
