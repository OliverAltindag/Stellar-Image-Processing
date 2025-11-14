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
    file_save(save_path, master_dark) 
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
    file_save(save_path, master_flat) 
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
    file_save(save_path, reduced_image, header)
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
    file_save(save_path, final_median_image, fits.getheader(list_image_paths[0]))
    return final_median_image

def reduction(data_folder_path, science_images_folder):
    '''
    '''
    standard_images_subfolder = "standard"
    standard_folder_path = os.path.join(data_folder_path, standard_images_subfolder)

    bias_files = filelist_creator(data_folder_path, 'calibration/biasframes')
    dark_files = filelist_creator(data_folder_path, 'calibration/darks')
    visual_flat_files = filelist_creator(data_folder_path, 'calibration/flats/visual')
    blue_flat_files = filelist_creator(data_folder_path, 'calibration/flats/blue')
    red_flat_files = filelist_creator(data_folder_path, 'calibration/flats/red')
    
    master_bias_path = os.path.join(data_folder_path, 'calibration/biasframes/master_bias.fits')
    master_bias(bias_files, master_bias_path)
    
    master_dark_path = os.path.join(data_folder_path, 'calibration/darks/master_dark.fits')
    master_dark(dark_files, master_bias_path, master_dark_path)

    master_flats_save_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    os.makedirs(master_flats_save_folder, exist_ok=True) 
    flat_file_groups = {
        'visual': visual_flat_files,
        'blue': blue_flat_files,
        'red': red_flat_files
    }
    for filter_name, file_list in flat_file_groups.items():
        save_path = os.path.join(master_flats_save_folder, f'master_flat_{filter_name}.fits')
        master_flat(file_list, master_bias_path, master_dark_path, save_path)  
    master_flats_folder = os.path.join(data_folder_path, 'calibration/flats/masters')

    science_folder_path = os.path.join(data_folder_path, science_images_folder)
    for filter_name in flat_file_groups.keys():
        filter_subfolder_path = os.path.join(science_folder_path, filter_name)
        science_files_raw = glob.glob(os.path.join(filter_subfolder_path, '*.fits'))
        for image_path in science_files_raw:
            if "fdb_" in image_path:
                continue
                
            base_filename = os.path.basename(image_path)
            new_filename = "fdb_" + base_filename
            final_save_path = os.path.join(filter_subfolder_path, new_filename)
            image_processing(image_path, master_bias_path, master_dark_path, master_flats_folder, final_save_path)
            
    standard_folder_path = os.path.join(data_folder_path, standard_images_subfolder)
    for filter_name in flat_file_groups.keys():
        filter_subfolder_path = os.path.join(standard_folder_path, filter_name)
        standard_files_raw = glob.glob(os.path.join(filter_subfolder_path, '*.fits'))
        for image_path in standard_files_raw:
            if "fdb_" in image_path:
                continue
            base_filename = os.path.basename(image_path)
            new_filename = "fdb_" + base_filename
            final_save_path = os.path.join(filter_subfolder_path, new_filename)
            image_processing(image_path, master_bias_path, master_dark_path, master_flats_folder, final_save_path)

    star_coords_main = [450, 470, 300, 320] # prettyy pretyyy good: guessed
    bg_coords_main = [480, 500, 280, 300]   
    pad_val = 100 

    # Run alignment and stacking for the target
    filter_files_main, all_shifts_main = sort_and_align_files(science_folder_path, star_coords_main, bg_coords_main)
    stack_all_filters(science_folder_path, filter_files_main, all_shifts_main, pad_val)

    star_coords_std = [123, 143, 205, 225] # random
    bg_coords_std = [150, 170, 150, 170]   

    filter_files_std, all_shifts_std = sort_and_align_files(standard_folder_path, star_coords_std, bg_coords_std)
    stack_all_filters(standard_folder_path, filter_files_std, all_shifts_std, pad_val)

    master_stack_paths = glob.glob(os.path.join(science_folder_path, "master_stack_*.fits"))
    ref_filter_name = 'blue' 
    master_ref_path = os.path.join(science_folder_path, f"master_stack_{ref_filter_name.lower()}.fits")
    
    if not os.path.exists(master_ref_path):
        master_ref_path = master_stack_paths[0]
        ref_filter_name = os.path.basename(master_ref_path).split('_')[-1].split('.')[0]

    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []

    for stack_path in master_stack_paths:
        files_to_align.append(stack_path)
        
        if stack_path == master_ref_path:
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            shifts = centroiding(stack_path, master_ref_path, star_coords_main, bg_coords_main)
            master_shifts_x.append(shifts[0])
            master_shifts_y.append(shifts[1])
                
    for i, stack_path in enumerate(files_to_align):
        base_name = os.path.basename(stack_path)
        aligned_save_path = os.path.join(science_folder_path, f"aligned_{base_name}")
        shifting([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val, aligned_save_path)
    return
