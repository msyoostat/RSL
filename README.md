# RSL
This repository includes the data and code used in the paper, **'Uncertainty-enabled machine learning for emulation of regional sea-level change caused by the Antarctic Ice Sheet'** by Yoo et al (2024). For the preprint, please check [here](https://arxiv.org/abs/2406.17729).

## Data
All data except the "full" dataset is stored in the **data** folder in this repository. The "full" dataset (rsl_full_data.Rdata) can be downloaded from the **link**.
- **Link**:
  - rsl_full_data.Rdata: The "full" data used in Section 1 to 6 can be downloaded from [here](https://drive.google.com/file/d/1ju48Dh3kfWOd1dqQmtAmBU-75Kw0elev/view?usp=sharing).
- **full**: This folder includes the reduced dimension input for the Gaussian process and random forest (principal components), as described in Section 3.3. The dimension reduction is conducted in Python and saved as .txt files for use in gp_cities.R.
    - train_data_x.txt: The reduced dimension input for training data.
    - val_data_x.txt: The reduced dimension input for validation data.
    - test_data_x.txt: The reduced dimension input for test data.
- **reduced**: This folder includes the data used in Appendix A. Note that .txt files are used only for gp_cities_reduced_input.R.
    - regional_cities_train_test_val_py.RData: training, validation, and test data used for the analysis in Appendix A.
    - train_data_lithk_region_mean_x.txt: input for training data.
    - val_data_lithk_region_mean_x.txt: input for validation data.
    - test_data_lithk_region_mean_x.txt: input for test data.
  
## Code for the machine learning (ML) models
All code is stored in the '**script**' folder in this repository.
- **full**: This folder includes the scripts for the analysis in Section 3.
  - cvae.py: python code for the CVAE.
  - fnn.py: python code for the feedforward NN.
  - rf.py: python code for the random forest.
  - gp_cities.R: R code for the Gaussian process.
- **reduced**: This folder includes the scripts for the analysis in Appendix A.
  - cvae_reduced_input.py: python code for the CVAE.
  - fnn_reduced_input.py: python code for the feedforward NN.
  - rf_reduced_input.py: python code for the random forest.
  - gp_cities_reduced_input.R: R code for the Gaussian process.
- **analysis**: This folder includes the scripts for the results in Section 5 of the paper. Note that gp_cities.R and gp_cities_reduced_input.R in the 'full' and 'reduced' folders, respectively, include the analysis code in it.
  - cvae_result.py: python code for the CVAE.
  - fnn_result.py: python code for the feedforward NN. It also includes code to generate Figure 6.
  - rf_result.py: python code for the random forest.
  - result_box_plot_final.R: R code to generate Figures 3, 4, 5, and 10.
  - result_box_plot_runtime_final.R: R code to generate Figure 7.
- **result**: This folder includes .csv files that store results.
  - result_table.csv: python code for the CVAE.
  - result_runtime.csv: python code for the feedforward NN.
  - result_reduced_table.csv: python code for the random forest.
