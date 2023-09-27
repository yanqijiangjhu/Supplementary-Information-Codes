# Supplementary-Information-Codes
Codes for network detection and analysis from paper _Hierarchical Assembly and Characterization of DNA Nanotube Networks Using Y-Shaped DNA Origami Seeds_

**Project Description**

The goal of this project is to investigate hierarchical network forming for origami-based Y-shaped seeds. The script ‘Simulation(Gillespie_algorithm).py’ may be run on individual files as it will output multiple data files. 

The main script in the repository, ‘Simulation(Gillespie_algorithm).py’, allows the users to compute joining rates for Y-shaped DNA nanotubes. The simulations are very useful as users can change diverse conditions to reflect their experimental design, including the number of nodes to be simulated, attributes of seed/nodes, conditions to allow joining, and the number of data files to be printed out by changing time steps. The script can also output the largest network size (seed size) after every joining rection. 

The general protocol of the script is as follows: compute pairwise joining rate for possible joining and sum up to calculate the total reaction rate. Perform joining based on pairwise joining rate, and the newly joined network is updated as one network. Then, the matrix containing joining rate elements is updated. The simulation runs until the total reaction rate becomes 0, that being said all possible joining has occurred. Further explanations of the simulation and each function can be found within the Python file itself.

'Gillespie_algorithm_simulation.ipynb' is the script that generates the joining reaction. After running the program, it will generate a bunch of files named after 'sim_(iteration)_simtime_(simulation time)_network_size_count.dat' which contain all the networks with their sizes (number of seeds) at a specific time. (It takes a long time to run!)

Output files in project directory: simulation_time.dat, average_network_size_of_a_seed.dat, network_size_count.dat (for each time step being printed out), largest_network_tracking.dat

**Additional packages to install before running**

This project relies on a number of Python packages, including pd Prior to running the Python files, users may need to install the Networkx package (https://networkx.org/documentation/stable/install.html). Note that these Python codes are designed to be run with Python version 3.

**Note**

In the simulation, we used a different expression to represent the K joining rate - we used K_join_fact instead. To make it clear, the 'fact(or)' means the additional multiplying factor to the cited K joining rate of 3.86e6/M/s. For example, when K_join_fact is 0.5, it equals to a K joining rate of 1.93e6/M/s. In order to make the simulation free of long numbers, we used factors 0.05, 0.1, 0.25, 0.33, 0.5, 0.75, and 1.0 instead.
