import helper_functions as h
import main_functions as mf

# The actual image reduction and stacking is done

def reduction(data_folder_path, science_images_folder):
    '''
    '''
    standard_images_subfolder = "standard"
    standard_folder_path = os.path.join(data_folder_path, standard_images_subfolder)
    science_folder_path = os.path.join(data_folder_path, science_images_folder)

    bias_files = h.filelist_creator(data_folder_path, 'calibration/biasframes')
    dark_files = h.filelist_creator(data_folder_path, 'calibration/darks')
    visual_flat_files = h.filelist_creator(data_folder_path, 'calibration/flats/visual')
    blue_flat_files = h.filelist_creator(data_folder_path, 'calibration/flats/blue')
    red_flat_files = h.filelist_creator(data_folder_path, 'calibration/flats/red')
    
    master_bias_path = os.path.join(data_folder_path, 'calibration/biasframes/master_bias.fits')
    mf.master_bias(bias_files, master_bias_path)
    
    master_dark_path = os.path.join(data_folder_path, 'calibration/darks/master_dark.fits')
    mf.master_dark(dark_files, master_bias_path, master_dark_path)

    master_flats_save_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    os.makedirs(master_flats_save_folder, exist_ok=True) 
    flat_file_groups = {
        'visual': visual_flat_files,
        'blue': blue_flat_files,
        'red': red_flat_files
    }
    for filter_name, file_list in flat_file_groups.items():
        save_path = os.path.join(master_flats_save_folder, f'master_flat_{filter_name}.fits')
        mf.master_flat(file_list, master_bias_path, master_dark_path, save_path) 
    
    master_flats_folder = os.path.join(data_folder_path, 'calibration/flats/masters')
    filter_names = flat_file_groups.keys()

    h.process_images_in_folder(science_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)
    h.process_images_in_folder(standard_folder_path, filter_names, master_bias_path, master_dark_path, master_flats_folder)

    pad_val = 100 
    star_coords_main = [450, 470, 300, 320]
    bg_coords_main = [480, 500, 280, 300]
    h.align_and_stack_folder(science_folder_path, star_coords_main, bg_coords_main, pad_val)
    
    star_coords_std = [123, 143, 205, 225]
    bg_coords_std = [150, 170, 150, 170]
    h.align_and_stack_folder(standard_folder_path, star_coords_std, bg_coords_std, pad_val)

    master_stack_paths = glob.glob(os.path.join(science_folder_path, "master_stack_*.fits"))
    ref_filter_name = 'blue' 
    master_ref_path = os.path.join(science_folder_path, f"master_stack_{ref_filter_name.lower()}.fits")

    master_shifts_x = []
    master_shifts_y = []
    files_to_align = []
    for stack_path in master_stack_paths:
        files_to_align.append(stack_path)
        if stack_path == master_ref_path:
            master_shifts_x.append(0.0)
            master_shifts_y.append(0.0)
        else:
            shifts = mf.centroiding(stack_path, master_ref_path, star_coords_main, bg_coords_main)
            master_shifts_x.append(shifts[0])
            master_shifts_y.append(shifts[1])
            
    for i, stack_path in enumerate(files_to_align):
        base_name = os.path.basename(stack_path)
        aligned_save_path = os.path.join(science_folder_path, f"aligned_{base_name}")
        mf.shifting([stack_path], [master_shifts_x[i]], [master_shifts_y[i]], pad_val, aligned_save_path)
    return
