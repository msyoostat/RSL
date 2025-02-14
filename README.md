# RSL
This repository includes the data and code used in the paper, **'Uncertainty-enabled machine learning for emulation of regional sea-level change caused by the Antarctic Ice Sheet'** by Yoo et al (2025). Specifically, this repository contains code for machine learning-based emulators. For the preprint, please check [here](https://arxiv.org/abs/2406.17729).

## Data
All data are openly available in [Yoo et al. (2025)](https://doi.org/10.5281/zenodo.14872314). 
- **sector_based**: This folder includes the data (sector-based inputs) used in **Appendix A**. Note that .txt files are used only for gp_cities_reduced_input.R.
    - regional_cities_train_test_val_py.RData: Training, validation, and test data.
    - train_data_lithk_region_mean_x.txt: The input for training data.
    - val_data_lithk_region_mean_x.txt: The input for validation data.
    - test_data_lithk_region_mean_x.txt: The input for test data.
- **main_result**: All other files in the data folder contain the data used for the main result in the paper.
  
## Script
All code for training models and analysis is stored in the '**script**' folder in this repository. Note that these are also archived in [Yoo et al. (2025)](https://doi.org/10.5281/zenodo.14872314). The script for the sector-based inputs is the same as the one used for the main results.
  - cvae.py: Python code for the CVAE.
  - fnn.py: Python code for the feedforward NN.
  - rf.py: Python code for the random forest.
  - gp_cities.R: R code for the Gaussian process.
- **sector_based**: This folder includes the scripts for training models in **Appendix A**.
  - cvae_reduced_input.py: Python code for the CVAE.
  - fnn_reduced_input.py: Python code for the feedforward NN.
  - rf_reduced_input.py: Python code for the random forest.
  - gp_cities_reduced_input.R: R code for the Gaussian process.

## Analysis
This folder includes the scripts for the results in the paper. That is, it includes the code for generating figures in the paper and for comparing UQ methods.

## Result files
Result files are archived in [Yoo et al. (2025)](https://doi.org/10.5281/zenodo.14872314).
