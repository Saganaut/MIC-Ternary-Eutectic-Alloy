# MIC-Ternary-Eutectic-Alloy
project page files


In order to run the smart pipeline:

# Python Libraries
* Numpy
* Scikit Learn (>= version 0.16)
* Scipy

# Set up directories
1. Create a data folder if one does not already exist in the main directory.
2. Add a folder to that called "test" (i.e. data/test)
3. Edit copy_data_locally.py
Edit the top two directories. 
data_path should be your dropbox folder for the simulation data. 
new_path should be the path to this directory data/test

Run copy_data_locally.py (it copies the janky old format of matlab to your project directory)
Then you should be able to run the pipeline:
python smart_pipeline.py
