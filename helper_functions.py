import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Helper Function Definitons will all be defined here:

def file_save(save_path, data, header = None):
    '''
    Saves data as a new fits file, with default header set to none which is helpful when making master calibration frames. 
    
    Parameters:
    -----------
    save_path: String
        Path to the chosen save location
    data: Array
        Data array to save to new location
    header: astropy Header [optional, default=None]
        Header the file will be saved with. Leave as None to save the file with no header.

    Returns:
    -----------
    None    
    '''
    fits.writeto(save_path, data, header, overwrite = True) # sets overwrite to true, albeit the code must be run in one shot
    return

def filelist_creator(base_path, subfolder_path):
    '''
    Constructs a new file path and returns the list of files within it. Essentially replaces our usage of glob.glob.
    
    Parameters:
    ----------
    base_path: String
        Path to the root directory
    subfolder_path: String
        Path to the sub-directory
        
    Returns:
    ---------
    List
        List of the paths of the files at the created location
    '''
    #creates new file path
    file_path = os.path.join(base_path, subfolder_path, '*')
    #retrieves files with paths matching the created path
    file_list = glob.glob(file_path)
    return file_list
    
def mediancombine(filelist):
    '''
    Function that creates a median image stack of the input files. 

    Parameters
    ----------
    filelist: List
        List of fits files to create a median stacked image from

    Returns
    -------
    Array
        3D Array containing the data of the median stacked image
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

def bias_subtract(filename, path_to_bias): 
    '''
    The function works to subracts the master biases from a single frame. 
    Leverages later architecture and the notion that bias subtract is the first calculation step,
    making filename a more efficient input.

    Parameters:
    ----------
    filename: String
        The path to the file to subtract the master bias frame from.
    path_to_bias: String
        The path to the master bias file.
        
    Returns:
    -------
    Array
        3D array of the data of the bias subtracted image
    '''
    # gets image data
    data = fits.getdata(filename)
    # gets master bias data
    master = fits.getdata(path_to_bias)
    # subtracts master bias from input data
    bias_subtracted = data - master
    return bias_subtracted

def dark_subtract(data, master_dark_path, scale_multiple):
    '''
    The function works to subracts the master darks from a single frame. 
    It has an added input so it scales the normalized master bias frame depending.

    Parameters:
    ----------
    filename: Array
        The raw image data which will be subtracted, in array form.
    path_to_bias: String
        The path to the master dark file
    scale_multiple: Float
        Exposure time to scale the master dark frame with

    Returns:
    --------
    Array
        3D array containing the data of the dark subtracted image
    '''
    # master dark data
    master_dark = fits.getdata(master_dark_path)
    # performs the subtract here, with the scalar multiple.
    dark_subtracted = data - (master_dark * scale_multiple)
    return dark_subtracted

def normalization(data, file_path):
    '''
    Finds the exposure time and normalizes the given data with it.
    
    Parameters
    ----------
    data: Array
        Array of image data 
    file_path: String
        Path to the file of the data

    Returns:
    --------
    Array
        Array of data normalized with the exposure time found in the file
    '''
    # gets the exposure time of the file
    header = fits.getheader(file_path)
    exptime = header["EXPTIME"]
    # normalizes the data here
    normalized = data / exptime
    return normalized

def flat_correct(science_data, master_flats_folder, filter_image):
    '''
    The function performs flat-field correction on a single science image. So will need to loop over them after. 

    Parameters:
    -----------
    science_data: Array
        The data array of the bias and dark subtracted science image to correct
    master_flat_path: String
        The path to the master flat file
    filter_image: String
        Filter used for the input image data
    
    Returns:
    --------
    Array
        3D array containing data of the flat corrected image
    '''
    # adaptive depending on the input filter, and gets the path
    flat_filename = f"master_flat_{filter_image.strip().lower()}.fit"
    master_flat_path = os.path.join(master_flats_folder, flat_filename)
    # Get the master flat data
    master_flat_data = fits.getdata(master_flat_path)
    master_flat_data[master_flat_data <= 0] = np.nan # if not there this did not work properly and would be an error :(
    # Divide the science image by the master flat, known as flat-fielding
    flat_corrected = science_data / master_flat_data
    return flat_corrected

def box_maker(image_path, star_coords, background_coords):
    '''
    Creates cutouts of a star and background patch within an image. 
    Used in centroiding to make the star boxes and background boxes.
    
    Parameters:
    -----------
    image_path: String
        Path to the image file
    star_coords: List
        List containing the minimum and maximum x and y values to use as the dimensions of the box
    background_coords
        List containing the minimum and maximum x and y values to use as the dimensions of the box

    Returns:
    --------
    Array
        The array of data from the star cutout box
    Array
        The array of data from the background cutout box
    Float
        minimum x-value of the star box
    Float
        minimum y value of the star box
    '''
    # gets the data and makes a copy to avoid errors later on
    original_data = fits.getdata(image_path)
    image_data = original_data.copy() 
    # unpacks the coordinates
    star_xmin, star_xmax, star_ymin, star_ymax = star_coords[0], star_coords[1], star_coords[2], star_coords[3]
    bg_xmin, bg_xmax, bg_ymin, bg_ymax = background_coords[0], background_coords[1], background_coords[2], background_coords[3]
    # makes the boxes with the above defined numbers
    star_box = image_data[int(star_ymin):int(star_ymax), int(star_xmin):int(star_xmax)]
    background_box = image_data[int(bg_ymin):int(bg_ymax), int(bg_xmin):int(bg_xmax)]
    return star_box, background_box, star_xmin, star_ymin

def sigma_finder(box):
    '''
    Finds the three sigma cutoff value to use when filtering data. 
    Important in star isolation.
    
    Parameters:
    ----------
    box: Array
        Data of the background cutout image.
        
    Returns:
    --------
    Float
       Cutoff value used to filter out background values from an image
    Float
        Average of the values in the background patch
    Float
        Standard deviation of the values in the background patch
        
    '''
    # gets the cutoff values using statistical values
    avg_background = np.mean(box)
    std_background = np.std(box)
    three_sigma_cutoff = 3 * std_background + avg_background
    return three_sigma_cutoff, avg_background, std_background

def star_isolator(box, cutoff, avg_background):
    '''
    Isolates a star within an image by setting any value less than the cutoff to NaN, which removes the background data.
    Important in centroiding. 

    Parameters:
    -----------
    box: Array
        Data of the cutout star image
    cutoff: Float
        Cutoff value to filter out the background with
    avg_background: Float
        Average value of the background cutout

    Returns:
    --------
    Array
        Array of image data after filtering out the background
    '''
    # sets any value less than the cutoff to NaN
    star = box <= cutoff
    box -= avg_background
    # removes the lower basis
    box[star] = np.nan
    return box

def star_finder(box, xmin, ymin):
    '''
    Finds the coordinates of the center of a star using the centroiding method.
    
    Parameters:
    -----------
    box: Array
        Data of the cutout star image
    xmin: Float
        Value of the minimum x-coordinate of the cutout box
    ymin: Float
        Value of the minimum y-coordinate of the cutout box
        
    Returns:
    --------
    Float
        The x-coordinate of the centroid found in the image
    Float
        The y-coordinate of the centroid found in the image
    '''
    # get image shape
    num_rows, num_cols = box.shape
    y_indices, x_indices = np.indices((num_rows, num_cols))
    # calculates sum of star pixel values
    sum_I = np.nansum(box)
    # weighted position sums
    x_sum_num = np.nansum(x_indices * box)
    y_sum_num = np.nansum(y_indices * box)
    # finds coordinates of centroid
    x_coord = xmin + (x_sum_num / sum_I)
    y_coord = ymin + (y_sum_num / sum_I)
    return x_coord, y_coord

def crop_center(img, target_rows, target_cols):
    '''
    Creates a window into the image which matches the size of the smallest dimensions between them
    centered in the image.
    
    Parameters:
    -----------
    img: Array
        Data of the image in array form
    target_rows: Float
        The number of rows the cutout will create
    target_cols: Float
        the number of rows the cutout will create
        
    Returns:
    --------
    img: Array
        The window of the image, not the entire image
    '''
    current_rows, current_cols = img.shape
    start_row = (current_rows - target_rows) // 2
    start_col = (current_cols - target_cols) // 2
    return img[start_row : start_row + target_rows, start_col : start_col + target_cols]
