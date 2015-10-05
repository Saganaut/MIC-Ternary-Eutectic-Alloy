meta_data = csvread('~/Documents/repos/project8883code/data/simulation_parameters.csv', 1, 0);
scatter3(meta_data(:, 1), meta_data(:, 3), meta_data(:, 5)) 
disp meta_data
