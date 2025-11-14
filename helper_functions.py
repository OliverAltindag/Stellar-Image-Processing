# Helper Function Definitons will all be defined here:

def file_save(save_path, data, header = None):
    '''
    '''
    fits.writeto(save_path, data, header, overwrite = True)
    return
    
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

def filelist_ceator(base_path, subfolder_path):
    '''
    '''
    file_path = os.path.join(base_path, subfolder_path, '*')
    file_list = glob.glob(file_path)
    return file_list
