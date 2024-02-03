import optims as OptiMS
import time

# Input output file/folder locations
software = 'MassLynx'
pic_folder = r''  # input folder location for saving pictures as r'folder_location\folder_name' if desired - else put False or r''
output_file = r'C:\Users\Waters\Documents\OptiMS\OptiMS_test'  # input location of optimisation file output as r'file_location\name' - do not include file extension, e.g. C:\Users\Waters\Documents\OptiMS\MS_opti_output.

# Input system times
scan_time = 1  # input acquisition time for each spectrum
scan_num = 8  # input number of spectrum scans required for each parameter configuration
stabil_delay = 2  # input time taken for system to stabilise in seconds (req. for run_optims and mz_grab only)
hold_end = 16  # input number of scans desired to hold optimal parameters immediately before termination (req. for run_optims and optims_grab only)
time_delay = 0  # insert a time delay between operations if MassLynx is running slow in format (req. for chrom_grab and mz_grab only)

# Input run_optims parameters
get_optims = True

# Input optimisation method details
method_type = 'BO'  # input method type: 'defined' for chosen parameter values; 'exhaustive' for exhaustive; 'random' for random combination from exhaustive; 'OFAT' for OFAT; 'BO' for BO; 'recover' to export last data set
method_metric = 'max'  # input metric defition: 'max' for maximising intensity of first chromatogram; 'ratio' for maximising intensity of first chromatogram to second; 'user def' to define own metric (below)
n_random_points = 60  # input number of random combinations of parameters to guess, must be >0 (random and BO only).
n_honing_points = 60  # input number of subsequent honing points required, must be >0 (BO only).
method_break = 0  # input a factor for which the method will break and move onto the next parameter, if the new metric falls below method_break * previous (OFAT only).

# Input parameter details, including bounds
tab_names = ['ES', 'Instrument']  # input names of param_in_tab in form ['Tab 1', ..., 'Tab X'] or None or [] if only tab used.
tab_rows = []  # input row of each tab in form [1, ..., 2] or None or [] if all param_in_tab on same row. Only two rows are currently supported.
param_names = ['Capillary', 'Sampling Cone', 'Source Offset', 'Nebuliser', 'TrapCE', 'TransCE']  # input names of parameters in form ['Param 1, ..., Param X']
default_params = [2.5, 0, 0, 2.5, 3, 0]  # input default start values in the form [0, ..., 0].
param_bounds = [(0, 5, 0.1), (0, 100, 1), (0, 100, 1), (2.5, 4, 0.1), (0, 20, 0.1), (0, 10, 0.1)]  # input parameters bounds (lower bound, upper bound, increment value) in form e.g. [(X1, X2, X3), ..., (Y1, Y2, Y3)].
param_in_tab = [1, 1, 1, 1, 2, 2]  # enter param_in_tab required for each parameter in form [1, ..., X] or None or [] if all parameters in Tab 1.
chrom_num = 3  # enter number of chromatograms in use. Input 0 if chromatograms cannot be copied. Else input integer > 0.

learn_coord = False  # input True if wanting to learn coordinates of required param_in_tab/boxes/chromatograms, else False.

# Define optims_grab parameters
get_optims_grab = False  # Set true if wanting to grab optims output for other species, else False
optims_import_file = r'C:\Users\Waters\Documents\PJHW\CBD_THC\Tandem MS references and tests\PJHW23022301_CBD_100UM_TRANSCE_0-30V_MS_opti_output.xlsx'  # Input filename of previous OptiMS output file, including extension, eg: import_file = r'C:\Users\IanC\Documents\Experiments\Exp_optims_output.xlsx'
optims_ranges = [(315.1824, 315.2824), (313.1667, 313.2667), (299.1511, 299.2511), (297.1718, 297.2718), (287.1875, 287.2875), (287.1511, 287.2511), (279.1613, 279.2613), (273.1354, 273.2354), (272.1276, 272.2276), (271.1198, 271.2198), (269.1769, 269.2769), (269.1405, 269.2405), (259.1198, 259.2198), (257.1041, 257.2041), (255.1613, 255.2613), (255.1249, 255.2249), (247.1198, 247.2198), (246.112, 246.212), (245.1405, 245.2405), (245.1041, 245.2041), (243.1249, 243.2249), (243.0885, 243.1885), (241.1092, 241.2092), (235.1198, 235.2198), (233.1041, 233.2041), (231.1249, 231.2249), (231.0885, 231.1885), (229.1092, 229.2092), (227.13, 227.23), (227.0936, 227.1936), (225.0779, 225.1779), (221.1041, 221.2041), (219.1249, 219.2249), (219.0885, 219.1885), (217.1092, 217.2092), (217.0728, 217.1728), (215.0936, 215.1936), (215.0572, 215.1572), (213.1143, 213.2143), (213.0779, 213.1779), (211.0623, 211.1623), (207.0885, 207.1885), (205.1092, 205.2092), (205.0728, 205.1728), (203.0936, 203.1936), (203.0572, 203.1572), (201.1143, 201.2143), (201.0779, 201.1779), (201.0415, 201.1415), (199.0987, 199.1987), (197.0466, 197.1466), (193.1456, 193.2456), (193.1092, 193.2092), (193.0728, 193.1728), (191.0936, 191.1936), (191.0572, 191.1572), (189.1143, 189.2143), (189.0779, 189.1779), (189.0415, 189.1415), (187.0987, 187.1987), (187.0259, 187.1259), (185.083, 185.183), (185.0466, 185.1466), (183.031, 183.131), (181.0728, 181.1728), (179.0936, 179.1936), (179.0572, 179.1572), (177.0779, 177.1779), (177.0415, 177.1415), (175.0987, 175.1987), (175.0623, 175.1623), (175.0259, 175.1259), (171.031, 171.131), (165.0415, 165.1415), (163.0623, 163.1623), (163.0259, 163.1259), (161.083, 161.183), (161.0466, 161.1466), (161.0103, 161.1103), (159.0674, 159.1674), (159.031, 159.131), (157.0517, 157.1517), (157.0153, 157.1153), (151.0623, 151.1623), (151.0259, 151.1259), (149.083, 149.183), (149.0466, 149.1466), (149.0103, 149.1103), (147.0674, 147.1674), (147.031, 147.131), (146.9946, 147.0946), (145.0517, 145.1517), (145.0153, 145.1153), (143.0361, 143.1361), (139.0623, 139.1623), (137.0466, 137.1466), (137.0103, 137.1103), (135.0674, 135.1674), (135.031, 135.131), (134.9946, 135.0946), (133.0517, 133.1517), (133.0153, 133.1153), (131.0361, 131.1361), (129.0204, 129.1204), (125.0466, 125.1466), (123.0674, 123.1674), (122.9946, 123.0946), (121.0517, 121.1517), (121.0153, 121.1153), (119.0361, 119.1361), (118.9997, 119.0997), (111.0674, 111.1674), (110.9946, 111.0946), (109.0517, 109.1517), (109.0153, 109.1153), (108.979, 109.079), (107.0361, 107.1361), (106.9997, 107.0997), (105.0204, 105.1204), (99.03099, 99.13099), (97.05172, 97.15172), (97.01534, 97.11534), (96.97895, 97.07895), (95.03607, 95.13607), (94.99969, 95.09969), (93.02042, 93.12042), (91.00477, 91.10477), (83.03607, 83.13607), (82.99969, 83.09969), (81.02042, 81.12042), (79.00477, 79.10477), (76.98912, 77.08912), (71.03607, 71.13607), (69.02042, 69.12042), (68.98404, 69.08404), (67.00477, 67.10477), (55.00477, 55.10477)]  # Input int number of chromatograms (e.g. 3) or list of int or tuples (e.g. [922.8727, (923.8727, 924.8776)]
optims_names = []  # Insert desired names for regions as list, e.g. ['Region 1', 'Region 2', ...] or leave empty for automated naming, i.e. [].

# Define chrom_grab parameters
get_chrom_grab = False  # Set true if wanting to grab multiple chromatograms, else False
chrom_ranges = optims_ranges  # Input list of mz values for extraction of chromatograms, where mz values are int (peak close to value) or tuple (range). E.g. [922.8727, (923.8727, 924.8776)]
chrom_names = []  # Insert desired names for regions as list, e.g. ['Region 1', 'Region 2', ...] or leave empty for automated naming, i.e. [].

# Define mz_grab parameters
get_mz_grab = False  # Set true if wanting to grab average mass spectra for certain chromatogram regions, else False
mz_ranges = r'C:\Users\Peter\Documents\Postdoctorate\Work\CBD to THC\Tandem MS\PJHW23022308_D9-THC_50UM_D8-THC_50UM_TRAPCE_0-30V_MS_opti_output.xlsx'  # Input list of tuples for ranges for averaging , e.g. [(0, 0.1), (0.2, 0.5), (1, 1.5)], or OptiMS output filename in form r'filename' or just 'output_file', using the filename from above
mz_names = [str(i) + 'V' for i in range(32)]  # Insert desired names for regions as list, e.g. ['Region 1', 'Region 2', ...] or leave empty for automated naming, i.e. [].

if get_optims:
    opti_param, opti_store_df = OptiMS.run_optims(param_names, param_bounds, default_params=default_params,
                                                  tab_names=tab_names, tab_rows=tab_rows, param_in_tab=param_in_tab,
                                                  method_type=method_type, method_metric=method_metric,
                                                  chrom_num=chrom_num, stabil_delay=stabil_delay,
                                                  scan_time=scan_time, scan_num=scan_num,
                                                  hold_end=hold_end, n_random_points=n_random_points,
                                                  n_honing_points=n_honing_points, break_fac=method_break,
                                                  learn_coord=learn_coord, software=software, pic_folder=pic_folder,
                                                  output_file=output_file, get_csv=False, get_Excel=True)

if get_optims and (get_optims_grab or get_chrom_grab or get_mz_grab):
    time.sleep(5)

if get_optims_grab:
    output_df = OptiMS.optims_grab(optims_import_file, optims_ranges, stabil_delay=stabil_delay, scan_num=scan_num,
                                   hold_end=hold_end, names=optims_names, software=software, learn_coord=learn_coord,
                                   time_delay=time_delay)
    OptiMS.optims_grab_process(output_df, optims_import_file)

if get_chrom_grab:
    raw_df, norm_df = OptiMS.chrom_grab(chrom_ranges, time_delay=time_delay)
    OptiMS.chrom_grab_process(raw_df, norm_df, output_file, get_csv=True, get_excel=True, get_pic=False)

if get_mz_grab:
    data = OptiMS.mz_grab(mz_ranges, stabil_delay=stabil_delay, scan_time=scan_time, scan_num=scan_num,
                          hold_end=hold_end, names=mz_names, time_delay=time_delay)
    OptiMS.mz_grab_process(data, output_file, get_csv=False, get_excel=True, get_pic=False)
