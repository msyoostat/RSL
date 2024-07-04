# RSL
This repository includes the data and code used in the paper, **'Uncertainty-enabled machine learning for emulation of regional sea-level change caused by the Antarctic Ice Sheet'** by Yoo et al (2024). For the preprint, please check [here](https://arxiv.org/abs/2406.17729).



## Data download
The data used in this work can be downloaded from [here](https://drive.google.com/file/d/1ju48Dh3kfWOd1dqQmtAmBU-75Kw0elev/view?usp=sharing). The data is stored in the .RData format and can be loaded in Python using the [pyreadr package](https://github.com/ofajardo/pyreadr).

## Code for the machine learning (ML) models.
All code is stored in the '**script**' folder in this repository.
- full: This folder includes scripts for the analysis in Section 3 of the paper.
  - **cvae.py**: python code for the CVAE.
  - **fnn.py**: python code for the feedforward NN.
  - **rf.py**: python code for the random forest.
  - **gp_cities.R**: R code for the Gaussian process.
- reduced: This folder includes scripts for the analysis in Appendix A of the paper.
  - **cvae_reduced_input.py**: python code for the CVAE.
  - **fnn_reduced_input.py**: python code for the feedforward NN.
  - **rf_reduced_input.py**: python code for the random forest.
  - **gp_cities_reduced_input.R**: R code for the Gaussian process.
- Item 3
