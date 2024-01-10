# Supplementary-Information-Codes
Codes for network detection and analysis from paper _Hierarchical Assembly and Characterization of DNA Nanotube Networks Using Y-Shaped DNA Origami Seeds_

**Project Description**

***1. Network growth simulation***

The goal of this project is to investigate hierarchical network forming for origami-based Y-shaped seeds. The script ‘Gillespie_algorithm_simulation.ipynb’ may be run on individual files as it will output multiple data files. 

The main script in the repository, ‘Gillespie_algorithm_simulation.ipynb’, allows the users to compute joining rates for Y-shaped DNA nanotubes. The simulations are very useful as users can change diverse conditions to reflect their experimental design, including the number of nodes to be simulated, attributes of seed/nodes, conditions to allow joining, and the number of data files to be printed out by changing time steps. The script can also output the largest network size (seed size) after every joining rection. 

The general protocol of the script is as follows: compute pairwise joining rate for possible joining and sum up to calculate the total reaction rate. Perform joining based on pairwise joining rate, and the newly joined network is updated as one network. Then, the matrix containing joining rate elements is updated. The simulation runs until the total reaction rate becomes 0, that being said all possible joining has occurred. Further explanations of the simulation and each function can be found within the Python file itself.

1. 'Gillespie_algorithm_simulation.ipynb' is the main script that sets up the reaction system, generates the joining reaction, and outputs several files that are needed by plotting. After running the program, it will generate a bunch of files named after 'sim_(iteration)_simtime_(simulation time)_network_size_count.dat' which contain all the networks with their sizes (number of seeds) at a specific time. (It takes a long time to run!)

Output files in project directory: simulation_time.dat, average_network_size_of_a_seed.dat, network_size_count.dat (for each time step being printed out), largest_network_tracking.dat

2. 'make_histograms.ipynb' is the script to generate network size distribution and print out the txt files of experimental results and simulated networks sizes. It also plots the distribution of network sizes at each time (1, 8, 23 hour(s)) using different K_joining.

Output files: data files'(x)hr_kjoinfact_(x)_50iters_to(x)seeds_simdata.txt', '(x)hr_kjoinfact_(x)_50iters_to(x)seeds_expdata.txt', plots '(x)hr_kjoinfact_(x)_50iters_to(x)seeds_plot.pdf'

3. 'collect_timepoint_data.ipynb' is the script to transform the simulation time generated by the scripts above to real-time data. And the detailed calculation and description are in ESI Section 3.1.

To make it more straight for Figure 6 and Figure 7 generations, we reorganized the files and in sub-folders of 'Figure 6/7_codes and input', the Figures 6 and 7 in the paper could be generated directly.

Additional packages to install before running:

This project relies on a number of Python packages, including pd prior to running the Python files, users may need to install the Networkx package (https://networkx.org/documentation/stable/install.html). Note that these Python codes are designed to be run with Python version 3.

**Note**

In the simulation, we used a different expression to represent the K_joining - we used K_join_fact instead. To make it clear, the 'fact(or)' means the additional multiplying factor to the cited K_joining of 3.86e6/M/s. For example, when K_join_fact is 0.5, it equals to a K_joining of 1.93e6/M/s. In order to make the simulation free of long numbers, we used factors 0.05, 0.1, 0.25, 0.33, 0.5, 0.75, and 1.0 instead.

***2. Edge and blob detection***

The script 'network_metrics.py' is built to detect the seeds in the networks from micrographs. This script is for visualizing the effect of Otsu's thresholding on an image. It shows the original image, its pixel intensity histogram with the threshold value, and the resulting binary image after thresholding.

To use this script, simply replace "image" in io.imread("image") with the path to your image file. The script will then perform the following steps:

1. Read the specified image.
2. Compute Otsu's threshold for the image.
3. Generate a binary image based on the threshold.
4. Display the original image, its histogram with the threshold value, and the binary image.

The script will display a window with three subplots:
1. The first subplot shows the original image.
2. The second subplot displays the histogram of the image with the calculated threshold.
3. The third subplot shows the resulting binary image after applying the threshold.
