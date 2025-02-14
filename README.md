# RSL
This repository includes the data and code used in the paper, **'Uncertainty-enabled machine learning for emulation of regional sea-level change caused by the Antarctic Ice Sheet'** by Yoo et al (2025). Specifically, this repository contains code for machine learning-based emulators. For the preprint, please check [here](https://arxiv.org/abs/2406.17729).

## Data
All data are openly available in [Yoo et al. (2025)](https://doi.org/10.5281/zenodo.14872314). 
- **full**: This folder includes the reduced dimension input for the Gaussian process and random forest (principal components), as described in Section 3.3. The dimension reduction is conducted in Python and saved as .txt files for use in gp_cities.R.
- **sector_based**: This folder includes the data (sector-based inputs) used in **Appendix A**. Note that .txt files are used only for gp_cities_reduced_input.R.
    - regional_cities_train_test_val_py.RData: Training, validation, and test data.
    - train_data_lithk_region_mean_x.txt: The input for training data.
    - val_data_lithk_region_mean_x.txt: The input for validation data.
    - test_data_lithk_region_mean_x.txt: The input for test data.
  
## Code
All code for training models and analysis is stored in the '**script**' folder in this repository.
- **full**: This folder includes the scripts for training models in **Sections 3 to 6**.
  - cvae.py: Python code for the CVAE.
  - fnn.py: Python code for the feedforward NN.
  - rf.py: Python code for the random forest.
  - gp_cities.R: R code for the Gaussian process.
- **sector_based**: This folder includes the scripts for training models in **Appendix A**.
  - cvae_reduced_input.py: Python code for the CVAE.
  - fnn_reduced_input.py: Python code for the feedforward NN.
  - rf_reduced_input.py: Python code for the random forest.
  - gp_cities_reduced_input.R: R code for the Gaussian process.
- **analysis**: This folder includes the scripts for the results in the paper. Note that gp_cities.R and gp_cities_reduced_input.R in the 'full' and 'sector_based' folders, respectively, include the analysis code in it.
  - cvae_result.py: Python code for the CVAE.
  - fnn_result.py: Python code for the feedforward NN. It also includes code to generate Figure 6.
  - rf_result.py: Python code for the random forest.
  - result_box_plot_final.R: R code to generate Figures 3, 4, 5, and 10.
  - result_box_plot_runtime_final.R: R code to generate Figure 7.
  - emulated_density_plots.Rmd: R code to generate Figure 8.

## Result

- **result**: This folder includes .csv and .RData files that store results.
  - result_table.csv: The summary of model performances for **Sections 3 to 6.**
  - result_reduced_table.csv: The summary of model performances for **Appendix A**.
  - result_runtime.csv: Run time summary.
  - emulated_data_fig8.RData: The emulated RSL change for Dunedin, Montevideo, and Midway during year 2100 with high-emissions scenario for **Section 6.**
