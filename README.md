# Supplementary-Information-Codes
Codes for network detection and analysis from paper _Hierarchical Assembly and Characterization of DNA Nanotube Networks Using Y-Shaped DNA Origami Seeds_

The goal of this project is to investigate hierarchical network forming for origami-based Y-shaped seeds. The scripts ‘Simulation(Gillespie_algorithm).py’ may be run on individual files as it will output multiple data files. 

The main script in the repository, ‘Simulation(Gillespie_algorithm).py’, allows the users to compute joining rates for Y-shaped DNA nanotubes. The simulations are very useful as users can change diverse conditions to reflect their experimental design, including the number of nodes to be simulated, attributes of seed/nodes, conditions to allow joining, and the number or data files to be printed out by changing time steps. The script can also output the largest network size (seed size) after every joining rection. 

The general protocol of the script is as follows: compute pairwise joining rate for possible joining and sum up to calculate total reaction rate. Perform joining based on pairwise joining rate, and the newly joined network is updated as one network. Then, the matrix containing joining rate elements is updated. The simulation runs until the total reaction rate becomes 0, that being said all possible joining has occurred. Further explanations of the simulation and each function can be found within the Python file itself.

Input files in project directory: Simulation(Gillespie_algorithm).py

Output files in project directory: simulation_time.dat, average_network_size_of_a_seed.dat, network_size_count.dat (for each time step being printed out), largest_network_tracking.dat

Additional packages to install before running

This project relies on a number of Python packages, including pd Prior to running the python files, users may need to install the Networkx package (https://networkx.org/documentation/stable/install.html). Note that these python codes are designed to be run with python version 3.
