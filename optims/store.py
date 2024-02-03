# Input optimisation method details
method_type = "hone"  # input method type: "defined" for chosen parameter values; "exhaust" for exhaustive; "random" for random combination from exhaustive; "simple" for simple; "hone" for honing; "recover" to export last data set
method_metric = "max"  # input metric defition: "max" for maximising intensity of first chromatogram; "ratio" for maximising intensity of first chromatogram to second; "user def" to define own metric (below)
method_break = ""  # input "y" or "yes" if requiring break clauses (simple only)
n_random_points = 150  # input number of random combinations of parameters to guess, must be >0 (random and hone only)
n_hone_honing_points = 60  # input number of subsequent honing points required, must be >0 (hone only)

# Input parameter details, including bounds
tab_names = ["ES", "Instrument"]  # input names of param_in_tab in form ["Tab 1", ..., "Tab X"]
tab_rows = []  # input row of each tab in form [1, ..., 2] or [] if all param_in_tab on same row. Only two rows are currently supported.
param_names = ["Capillary", "Sampling Cone", "Source Offset", "Nebuliser", "Trap CE", "Transfer CE"]  # input names of parameters in form ["Param 1, ..., Param X"]
default_params = [3, 10, 40, 3, 2, 0]  # input "default" to use default parameters as a starting value or default start values in the form [0, ..., 0] for the six (std. [3, 40, 80, 3, 4, 2])
param_bounds = [(0, 5, 0.1), (0, 100, 1), (0, 100, 1), (2.5, 3.5, 0.1), (0, 10, 0.1), (0, 10, 0.1)]  # input parameters bounds (lower bound, upper bound, increment value) in form e.g. [(X1, X2, X3), ..., (Y1, Y2, Y3)]
tabs = [1, 1, 1, 1, 2, 2]  # enter param_in_tab required for each parameter in form [1, ..., X] or [] if all parameters in Tab 1
chrom_num = 2  # enter number of chromatograms in use

box_coord = ""  # input "learn" or "?" if want to learn coordinates of required param_in_tab/boxes/chromatograms, "" if not

# Input system times
param_change_delay = 10  # input delay between change of parameters in seconds
stabil_delay = 2  # input time taken for system to stabilise in seconds
hold_end = 60  # input time desired to hold optimal parameters immediately before termination in seconds

# Input output file/folder locations
output_file = r"C:\Users\Waters\Documents\OptiMS\MS_opti_output"  # input location of optimisation file output as r"file_location\name" - do not include file extension C:\Users\Waters\Documents\OptiMS\MS_opti_output
input_file = r"C:\Users\Waters\Documents\OptiMS\MS_opti_output.xlsx"  # input file location of previously saved parameter details (defined only)
input_sheet = "Params"  # # input sheet name of previously saved parameter details (defined only)
pic_folder = r""  # input folder location for saving pictures as r"folder_location\folder_name" if desired - else leave as r""