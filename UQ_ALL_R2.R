#make sure to change working directory to where .RData files and kernel_ensemble and ML_output directories are located
#e.g. setwd(...)
load('val.NEW.RData')
load('test.NEW.RData')
load('cities.RData')

#1 UQ for kernel method
val_files <- read.csv('val_name.csv')
val_file_names <- strsplit(val_files[,2], split = "_")
n_cities <- 27
accs_kernel <- rep(0, n_cities)
rmse_kernel <- rep(0, n_cities)
coverages_kernel <- matrix(rep(0, n_cities*4), ncol = 4)
lengths_kernel <- matrix(rep(0, n_cities*4), ncol = 4)
int_scores_kernel <- matrix(rep(0, n_cities*4), ncol = 4)

kernel_pred_val <- matrix(rep(0, n_cities*119), ncol = n_cities)
kernel_pred_test <- matrix(rep(0, n_cities*119), ncol = n_cities)

for(city_count in 1:n_cities){
  cur_val <- rep(0, length(val_file_names))
  for(i in 1:length(val_file_names)){
    cur_names <- val_file_names[[i]]
    cur_names <- cur_names[-length(cur_names)]
    cur_names <- cur_names[-length(cur_names)]
    cur_name <- paste(cur_names, collapse = "_")
    cur_name <- paste(cur_name, ".txt", sep = '')
    cur_name <- paste('./kernel_ensemble/',cur_name,sep = '')
    print(cur_name)
    cur_dat <- read.csv(cur_name, header = FALSE)
    cur_val[i] <- cur_dat[city_count,1]
  }
  kernel_pred_val[, city_count] <- cur_val
  cur_val_sims <- val.data.y.NEW[,city_count]
  
  val_dat_cur <- data.frame(cur_val, cur_val_sims)
  names(val_dat_cur) <- c('pred', 'obs')
  val_lm <- lm(obs ~ pred, data = val_dat_cur)
  
  test_files <- read.csv('test_name.csv')
  test_file_names <- strsplit(test_files[,2], split = "_")
  
  cur_test <- rep(0, length(test_file_names))
  
  for(i in 1:length(test_file_names)){
    cur_names <- test_file_names[[i]]
    cur_names <- cur_names[-length(cur_names)]
    cur_names <- cur_names[-length(cur_names)]
    cur_name <- paste(cur_names, collapse = "_")
    cur_name <- paste(cur_name, ".txt", sep = '')
    cur_name <- paste('./kernel_ensemble/',cur_name,sep = '')
    print(cur_name)
    cur_dat <- read.csv(cur_name, header = FALSE)
    cur_test[i] <- cur_dat[city_count,1]
  }
  kernel_pred_test[, city_count] <- cur_test
  cur_test_sims <- test.data.y.NEW[,city_count]
  test_data_cur <- data.frame(cur_test)
  names(test_data_cur) <- c('pred')
  test_predictions <- predict(val_lm, newdata = test_data_cur)
  accs_kernel[city_count] <- cor(test_predictions, cur_test_sims, method = 'pearson')^2
  rmse_kernel[city_count] <- sqrt(mean((test_predictions-cur_test_sims)^2))
  
  test_predictions_85 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .85))
  test_predictions_90 <-  data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .90))
  test_predictions_95 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .95))
  test_predictions_99 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .99))
  
  coverage_85 <- rep(0, nrow(test_data_cur))
  coverage_90 <-  rep(0, nrow(test_data_cur))
  coverage_95 <-  rep(0, nrow(test_data_cur))
  coverage_99 <-  rep(0, nrow(test_data_cur))
  
  lengths_85 <- rep(0, nrow(test_data_cur))
  lengths_90 <-  rep(0, nrow(test_data_cur))
  lengths_95 <-  rep(0, nrow(test_data_cur))
  lengths_99 <-  rep(0, nrow(test_data_cur))
  
  interval_score_85 <- rep(0, nrow(test_data_cur))
  interval_score_90 <-  rep(0, nrow(test_data_cur))
  interval_score_95 <-  rep(0, nrow(test_data_cur))
  interval_score_99 <-  rep(0, nrow(test_data_cur))
  
  for(i in 1:nrow(test_data_cur)){
    if((test_predictions_85$upr[i] >= cur_test_sims[i]) && (test_predictions_85$lwr[i] <= cur_test_sims[i])){
      coverage_85[i] <- 1
    }
    if((test_predictions_90$upr[i] >= cur_test_sims[i]) && (test_predictions_90$lwr[i] <= cur_test_sims[i])){
      coverage_90[i] <- 1
    }
    if((test_predictions_95$upr[i] >= cur_test_sims[i]) && (test_predictions_95$lwr[i] <= cur_test_sims[i])){
      coverage_95[i] <- 1
    }
    if((test_predictions_99$upr[i] >= cur_test_sims[i]) && (test_predictions_99$lwr[i] <= cur_test_sims[i])){
      coverage_99[i] <- 1
    }
    lengths_85[i] <- test_predictions_85$upr[i] - test_predictions_85$lwr[i] 
    lengths_90[i] <- test_predictions_90$upr[i] - test_predictions_90$lwr[i]
    lengths_95[i] <- test_predictions_95$upr[i] - test_predictions_95$lwr[i]
    lengths_99[i] <- test_predictions_99$upr[i] - test_predictions_99$lwr[i]
    
    #include proper interval score following Gneiting and Raftery (2007)
    interval_score_85[i] <- lengths_85[i]+(2/.15)*(test_predictions_85$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_85$lwr[i])+(2/.15)*(cur_test_sims[i] -test_predictions_85$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_85$upr[i])
    interval_score_90[i] <- lengths_90[i]+(2/.1)*(test_predictions_90$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_90$lwr[i])+(2/.1)*(cur_test_sims[i] -test_predictions_90$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_90$upr[i])
    interval_score_95[i] <- lengths_95[i]+(2/.05)*(test_predictions_95$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_95$lwr[i])+(2/.05)*(cur_test_sims[i] -test_predictions_95$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_95$upr[i])
    interval_score_99[i] <- lengths_99[i]+(2/.01)*(test_predictions_95$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_99$lwr[i])+(2/.01)*(cur_test_sims[i] -test_predictions_99$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_99$upr[i])
  }
  
  coverages_kernel[city_count,1] <- mean(coverage_85)
  coverages_kernel[city_count,2] <- mean(coverage_90)
  coverages_kernel[city_count,3] <- mean(coverage_95)
  coverages_kernel[city_count,4] <- mean(coverage_99)
  
  lengths_kernel[city_count, 1] <- mean(lengths_85)
  lengths_kernel[city_count, 2] <- mean(lengths_90)
  lengths_kernel[city_count, 3] <- mean(lengths_95)
  lengths_kernel[city_count, 4] <- mean(lengths_99)
  
  int_scores_kernel[city_count, 1] <- mean(interval_score_85)
  int_scores_kernel[city_count, 2] <- mean(interval_score_90)
  int_scores_kernel[city_count, 3] <- mean(interval_score_95)
  int_scores_kernel[city_count, 4] <- mean(interval_score_99)
}

#read in FNN predictions
fnn_pred_test <- as.matrix(read.csv('./ML_output/fnn_predictions_test.csv', header = FALSE))
fnn_pred_val <- as.matrix(read.csv('./ML_output/fnn_predictions_val.csv', header = FALSE))

#read in CVAE predictions
CVAE_ensemble <- array(rep(0,27*119*500), dim=c(500,119,27))
for(i in 1:27){
  print(i)
  cur_file <- paste('./ML_output/',i,'cvae_ensemble.csv', sep = '')
  cur_mat <- as.matrix(read.csv(cur_file, header = FALSE))
  CVAE_ensemble[,,i] <- cur_mat
}
cvae_pred_val <- as.matrix(read.csv('./ML_output/cvae_val_pred.csv', header = FALSE))
cvae_pred_test <- matrix(rep(0,27*119), ncol = 27)
for(i in 1:27){
  cvae_pred_test[,i] <- colMeans(CVAE_ensemble[,,i])
}

#FNN
n_cities <- 27
accs_nn <- rep(0, n_cities)
rmse_nn <- rep(0, n_cities)
coverages_nn <- matrix(rep(0, n_cities*4), ncol = 4)
lengths_nn <- matrix(rep(0, n_cities*4), ncol = 4)
int_scores_nn <- matrix(rep(0, n_cities*4), ncol = 4)

for(city_count in 1:n_cities){
  print(city_count)
  cur_val <- fnn_pred_val[,city_count]
  cur_val_sims <- val.data.y.NEW[,city_count]
  
  val_dat_cur <- data.frame(cur_val, cur_val_sims)
  names(val_dat_cur) <- c('pred', 'obs')
  val_lm <- lm(obs ~ pred, data = val_dat_cur)
  
  test_files <- read.csv('test_name.csv')
  test_file_names <- strsplit(test_files[,2], split = "_")
  
  cur_test <- fnn_pred_test[,city_count]
  
  cur_test_sims <- test.data.y.NEW[,city_count]
  test_data_cur <- data.frame(cur_test)
  names(test_data_cur) <- c('pred')
  test_predictions <- predict(val_lm, newdata = test_data_cur)
  accs_nn[city_count] <- cor(test_predictions, cur_test_sims, method = 'pearson')^2
  rmse_nn[city_count] <- sqrt(mean((test_predictions-cur_test_sims)^2))
  
  plot_seq <- data.frame(seq(-.5, .5, length.out = 1000))
  names(plot_seq) <- c('pred')
  
  test_predictions_85 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .85))
  test_predictions_90 <-  data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .90))
  test_predictions_95 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .95))
  test_predictions_99 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .99))
  
  coverage_85 <- rep(0, nrow(test_data_cur))
  coverage_90 <-  rep(0, nrow(test_data_cur))
  coverage_95 <-  rep(0, nrow(test_data_cur))
  coverage_99 <-  rep(0, nrow(test_data_cur))
  
  lengths_85 <- rep(0, nrow(test_data_cur))
  lengths_90 <-  rep(0, nrow(test_data_cur))
  lengths_95 <-  rep(0, nrow(test_data_cur))
  lengths_99 <-  rep(0, nrow(test_data_cur))
  
  interval_score_85 <- rep(0, nrow(test_data_cur))
  interval_score_90 <-  rep(0, nrow(test_data_cur))
  interval_score_95 <-  rep(0, nrow(test_data_cur))
  interval_score_99 <-  rep(0, nrow(test_data_cur))
  
  for(i in 1:nrow(test_data_cur)){
    if((test_predictions_85$upr[i] >= cur_test_sims[i]) && (test_predictions_85$lwr[i] <= cur_test_sims[i])){
      coverage_85[i] <- 1
    }
    if((test_predictions_90$upr[i] >= cur_test_sims[i]) && (test_predictions_90$lwr[i] <= cur_test_sims[i])){
      coverage_90[i] <- 1
    }
    if((test_predictions_95$upr[i] >= cur_test_sims[i]) && (test_predictions_95$lwr[i] <= cur_test_sims[i])){
      coverage_95[i] <- 1
    }
    if((test_predictions_99$upr[i] >= cur_test_sims[i]) && (test_predictions_99$lwr[i] <= cur_test_sims[i])){
      coverage_99[i] <- 1
    }
    lengths_85[i] <- test_predictions_85$upr[i] - test_predictions_85$lwr[i] 
    lengths_90[i] <- test_predictions_90$upr[i] - test_predictions_90$lwr[i]
    lengths_95[i] <- test_predictions_95$upr[i] - test_predictions_95$lwr[i]
    lengths_99[i] <- test_predictions_99$upr[i] - test_predictions_99$lwr[i]
    
    #include proper interval score following Gneiting and Raftery (2007)
    interval_score_85[i] <- lengths_85[i]+(2/.15)*(test_predictions_85$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_85$lwr[i])+(2/.15)*(cur_test_sims[i] -test_predictions_85$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_85$upr[i])
    interval_score_90[i] <- lengths_90[i]+(2/.1)*(test_predictions_90$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_90$lwr[i])+(2/.1)*(cur_test_sims[i] -test_predictions_90$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_90$upr[i])
    interval_score_95[i] <- lengths_95[i]+(2/.05)*(test_predictions_95$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_95$lwr[i])+(2/.05)*(cur_test_sims[i] -test_predictions_95$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_95$upr[i])
    interval_score_99[i] <- lengths_99[i]+(2/.01)*(test_predictions_95$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_99$lwr[i])+(2/.01)*(cur_test_sims[i] -test_predictions_99$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_99$upr[i])
  }
  
  coverages_nn[city_count,1] <- mean(coverage_85)
  coverages_nn[city_count,2] <- mean(coverage_90)
  coverages_nn[city_count,3] <- mean(coverage_95)
  coverages_nn[city_count,4] <- mean(coverage_99)
  
  lengths_nn[city_count, 1] <- mean(lengths_85)
  lengths_nn[city_count, 2] <- mean(lengths_90)
  lengths_nn[city_count, 3] <- mean(lengths_95)
  lengths_nn[city_count, 4] <- mean(lengths_99)
  
  int_scores_nn[city_count, 1] <- mean(interval_score_85)
  int_scores_nn[city_count, 2] <- mean(interval_score_90)
  int_scores_nn[city_count, 3] <- mean(interval_score_95)
  int_scores_nn[city_count, 4] <- mean(interval_score_99)
}

# CVAE
n_cities <- 27
accs_cvae <- rep(0, n_cities)
rmse_cvae <- rep(0, n_cities)
coverages_cvae <- matrix(rep(0, n_cities*4), ncol = 4)
lengths_cvae <- matrix(rep(0, n_cities*4), ncol = 4)
int_scores_cvae <- matrix(rep(0, n_cities*4), ncol = 4)

for(city_count in 1:n_cities){
  print(city_count)
  cur_val <- cvae_pred_val[,city_count]
  cur_val_sims <- val.data.y.NEW[,city_count]
  
  val_dat_cur <- data.frame(cur_val, cur_val_sims)
  names(val_dat_cur) <- c('pred', 'obs')
  val_lm <- lm(obs ~ pred, data = val_dat_cur)
  
  test_files <- read.csv('test_name.csv')
  test_file_names <- strsplit(test_files[,2], split = "_")
  
  cur_test <- cvae_pred_test[,city_count]
  cur_test_sims <- test.data.y.NEW[,city_count]
  test_data_cur <- data.frame(cur_test)
  names(test_data_cur) <- c('pred')
  test_predictions <- predict(val_lm, newdata = test_data_cur)
  accs_cvae[city_count] <- cor(test_predictions, cur_test_sims, method = 'pearson')^2
  rmse_cvae[city_count] <- sqrt(mean((test_predictions-cur_test_sims)^2))
  
  test_predictions_85 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .85))
  test_predictions_90 <-  data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .90))
  test_predictions_95 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .95))
  test_predictions_99 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .99))
  
  plot_seq <- data.frame(seq(-.5, .5, length.out = 1000))
  names(plot_seq) <- c('pred')
  
  test_predictions_85 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .85))
  test_predictions_90 <-  data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .90))
  test_predictions_95 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .95))
  test_predictions_99 <- data.frame(predict(val_lm, newdata = test_data_cur, interval = "predict", level = .99))
  
  coverage_85 <- rep(0, nrow(test_data_cur))
  coverage_90 <-  rep(0, nrow(test_data_cur))
  coverage_95 <-  rep(0, nrow(test_data_cur))
  coverage_99 <-  rep(0, nrow(test_data_cur))
  
  lengths_85 <- rep(0, nrow(test_data_cur))
  lengths_90 <-  rep(0, nrow(test_data_cur))
  lengths_95 <-  rep(0, nrow(test_data_cur))
  lengths_99 <-  rep(0, nrow(test_data_cur))
  
  for(i in 1:nrow(test_data_cur)){
    if((test_predictions_85$upr[i] >= cur_test_sims[i]) && (test_predictions_85$lwr[i] <= cur_test_sims[i])){
      coverage_85[i] <- 1
    }
    if((test_predictions_90$upr[i] >= cur_test_sims[i]) && (test_predictions_90$lwr[i] <= cur_test_sims[i])){
      coverage_90[i] <- 1
    }
    if((test_predictions_95$upr[i] >= cur_test_sims[i]) && (test_predictions_95$lwr[i] <= cur_test_sims[i])){
      coverage_95[i] <- 1
    }
    if((test_predictions_99$upr[i] >= cur_test_sims[i]) && (test_predictions_99$lwr[i] <= cur_test_sims[i])){
      coverage_99[i] <- 1
    }
    lengths_85[i] <- test_predictions_85$upr[i] - test_predictions_85$lwr[i] 
    lengths_90[i] <- test_predictions_90$upr[i] - test_predictions_90$lwr[i]
    lengths_95[i] <- test_predictions_95$upr[i] - test_predictions_95$lwr[i]
    lengths_99[i] <- test_predictions_99$upr[i] - test_predictions_99$lwr[i]
    
    #include proper interval score following Gneiting and Raftery (2007)
    interval_score_85[i] <- lengths_85[i]+(2/.15)*(test_predictions_85$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_85$lwr[i])+(2/.15)*(cur_test_sims[i] -test_predictions_85$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_85$upr[i])
    interval_score_90[i] <- lengths_90[i]+(2/.1)*(test_predictions_90$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_90$lwr[i])+(2/.1)*(cur_test_sims[i] -test_predictions_90$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_90$upr[i])
    interval_score_95[i] <- lengths_95[i]+(2/.05)*(test_predictions_95$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_95$lwr[i])+(2/.05)*(cur_test_sims[i] -test_predictions_95$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_95$upr[i])
    interval_score_99[i] <- lengths_99[i]+(2/.01)*(test_predictions_95$lwr[i] - cur_test_sims[i])*as.numeric(cur_test_sims[i] < test_predictions_99$lwr[i])+(2/.01)*(cur_test_sims[i] -test_predictions_99$upr[i])*as.numeric(cur_test_sims[i] > test_predictions_99$upr[i])
  }
  
  coverages_cvae[city_count,1] <- mean(coverage_85)
  coverages_cvae[city_count,2] <- mean(coverage_90)
  coverages_cvae[city_count,3] <- mean(coverage_95)
  coverages_cvae[city_count,4] <- mean(coverage_99)
  
  lengths_cvae[city_count, 1] <- mean(lengths_85)
  lengths_cvae[city_count, 2] <- mean(lengths_90)
  lengths_cvae[city_count, 3] <- mean(lengths_95)
  lengths_cvae[city_count, 4] <- mean(lengths_99)
  
  int_scores_cvae[city_count, 1] <- mean(interval_score_85)
  int_scores_cvae[city_count, 2] <- mean(interval_score_90)
  int_scores_cvae[city_count, 3] <- mean(interval_score_95)
  int_scores_cvae[city_count, 4] <- mean(interval_score_99)
}

#check coverage and length of CVAE ensemble
cvae_ensemble_coverage <- rep(0, 27)
cvae_ensemble_length <- rep(0, 27)
for(i in 1:27){
  print(i)
  cur_city_coverage <- rep(0, 119)
  cur_city_lengths <- rep(0, 119)
  for(j in 1:119){
    min_val <- min(CVAE_ensemble[,j,i])
    max_val <- max(CVAE_ensemble[,j,i])
    cur_city_lengths[j] <- max_val - min_val
    if((max_val >= test.data.y.NEW[j,i]) && (min_val <= test.data.y.NEW[j,i])){
      cur_city_coverage[j] <- 1
    }
  }
  cvae_ensemble_coverage[i] <- mean(cur_city_coverage)
  cvae_ensemble_length[i] <-  mean(cur_city_lengths)
}

#use split conformal for all three methods

#A: FNN
coverage_nn_conf <- matrix(rep(0,27*4), ncol = 4)
length_nn_conf <- matrix(rep(0,27*4), ncol = 4)
int_scores_nn_conf <- matrix(rep(0,27*4), ncol = 4)

for(i in 1:27){
  resids_cur <- abs(val.data.y.NEW[,i] - fnn_pred_val[,i])
  sorted_resids_cur <- sort(resids_cur)
  
  k_15 <- ceiling((119 + 1)*(1-.15))
  k_10 <- ceiling((119 + 1)*(1-.10))
  k_05 <- ceiling((119 + 1)*(1-.05))
  k_01 <- ceiling((119 + 1)*(1-.01))
  
  coverage_15 <- rep(0, 119)
  coverage_10 <- rep(0, 119)
  coverage_05 <- rep(0, 119)
  coverage_01 <- rep(0, 119)
  
  interval_score_85 <- rep(0, 119)
  interval_score_90 <- rep(0, 119)
  interval_score_95 <- rep(0, 119)
  interval_score_99 <- rep(0, 119)
  
  cur_test_sims <- test.data.y.NEW[,i]
  for(j in 1:119){
    if(((fnn_pred_test[j,i]+sorted_resids_cur[k_15]) >= cur_test_sims[j]) && ((fnn_pred_test[j,i]-sorted_resids_cur[k_15]) <= cur_test_sims[j])){
      coverage_15[j] <- 1
    }
    if(((fnn_pred_test[j,i]+sorted_resids_cur[k_10]) >= cur_test_sims[j]) && ((fnn_pred_test[j,i]-sorted_resids_cur[k_10]) <= cur_test_sims[j])){
      coverage_10[j] <- 1
    }
    if(((fnn_pred_test[j,i]+sorted_resids_cur[k_05]) >= cur_test_sims[j]) && ((fnn_pred_test[j,i]-sorted_resids_cur[k_05]) <= cur_test_sims[j])){
      coverage_05[j] <- 1
    }
    if(((fnn_pred_test[j,i]+sorted_resids_cur[k_01]) >= cur_test_sims[j]) && ((fnn_pred_test[j,i]-sorted_resids_cur[k_01]) <= cur_test_sims[j])){
      coverage_01[j] <- 1
    }
    
    length_85 <- 2*sorted_resids_cur[k_15]
    length_90 <- 2*sorted_resids_cur[k_10]
    length_95 <- 2*sorted_resids_cur[k_05]
    length_99 <- 2*sorted_resids_cur[k_01]
    
    cur_true <- fnn_pred_test[j,i]
    lwr_85 <- cur_true - sorted_resids_cur[k_15]
    upr_85 <- cur_true + sorted_resids_cur[k_15]
    lwr_90 <- cur_true - sorted_resids_cur[k_10]
    upr_90 <- cur_true + sorted_resids_cur[k_10]
    lwr_95 <- cur_true - sorted_resids_cur[k_05]
    upr_95 <- cur_true + sorted_resids_cur[k_05]
    lwr_99 <- cur_true - sorted_resids_cur[k_01]
    upr_99 <- cur_true + sorted_resids_cur[k_01]
    
    interval_score_85[j] <- length_85+(2/.15)*(lwr_85 - cur_true)*as.numeric(cur_true < lwr_85)+(2/.15)*(cur_true-upr_85)*as.numeric(cur_true > upr_85)
    interval_score_90[j] <- length_90+(2/.1)*(lwr_90 - cur_true)*as.numeric(cur_true < lwr_90)+(2/.1)*(cur_true -upr_90)*as.numeric(cur_true > upr_90)
    interval_score_95[j] <- length_95+(2/.05)*(lwr_95 - cur_true)*as.numeric(cur_true < lwr_95)+(2/.05)*(cur_true -upr_95)*as.numeric(cur_true > upr_95)
    interval_score_99[j] <- length_99+(2/.01)*(lwr_99 - cur_true)*as.numeric(cur_true < lwr_99)+(2/.01)*(cur_true -upr_99)*as.numeric(cur_true > upr_99)
  }
  print(i)
  coverage_nn_conf[i,1] <- mean(coverage_15)
  coverage_nn_conf[i,2] <- mean(coverage_10)
  coverage_nn_conf[i,3] <- mean(coverage_05)
  coverage_nn_conf[i,4] <- mean(coverage_01)
  length_nn_conf[i,1] <- 2*sorted_resids_cur[k_15]
  length_nn_conf[i,2] <- 2*sorted_resids_cur[k_10]
  length_nn_conf[i,3] <- 2*sorted_resids_cur[k_05]
  length_nn_conf[i,4] <- 2*sorted_resids_cur[k_01]
  int_scores_nn_conf[i,1] <- mean(interval_score_85)
  int_scores_nn_conf[i,2] <- mean(interval_score_90)
  int_scores_nn_conf[i,3] <- mean(interval_score_95)
  int_scores_nn_conf[i,4] <- mean(interval_score_99)
}

#B: CVAE
coverage_cvae_conf <- matrix(rep(0,27*4), ncol = 4)
length_cvae_conf <- matrix(rep(0,27*4), ncol = 4)
int_scores_cvae_conf <- matrix(rep(0,27*4), ncol = 4)

for(i in 1:27){
  resids_cur <- abs(val.data.y.NEW[,i] - cvae_pred_val[,i])
  sorted_resids_cur <- sort(resids_cur)
  
  k_15 <- ceiling((119 + 1)*(1-.15))
  k_10 <- ceiling((119 + 1)*(1-.10))
  k_05 <- ceiling((119 + 1)*(1-.05))
  k_01 <- ceiling((119 + 1)*(1-.01))
  
  coverage_15 <- rep(0, 119)
  coverage_10 <- rep(0, 119)
  coverage_05 <- rep(0, 119)
  coverage_01 <- rep(0, 119)
  
  cur_test_sims <- test.data.y.NEW[,i]
  for(j in 1:119){
    if(((cvae_pred_test[j,i]+sorted_resids_cur[k_15]) >= cur_test_sims[j]) && ((cvae_pred_test[j,i]-sorted_resids_cur[k_15]) <= cur_test_sims[j])){
      coverage_15[j] <- 1
    }
    if(((cvae_pred_test[j,i]+sorted_resids_cur[k_10]) >= cur_test_sims[j]) && ((cvae_pred_test[j,i]-sorted_resids_cur[k_10]) <= cur_test_sims[j])){
      coverage_10[j] <- 1
    }
    if(((cvae_pred_test[j,i]+sorted_resids_cur[k_05]) >= cur_test_sims[j]) && ((cvae_pred_test[j,i]-sorted_resids_cur[k_05]) <= cur_test_sims[j])){
      coverage_05[j] <- 1
    }
    if(((cvae_pred_test[j,i]+sorted_resids_cur[k_01]) >= cur_test_sims[j]) && ((cvae_pred_test[j,i]-sorted_resids_cur[k_01]) <= cur_test_sims[j])){
      coverage_01[j] <- 1
    }
    length_85 <- 2*sorted_resids_cur[k_15]
    length_90 <- 2*sorted_resids_cur[k_10]
    length_95 <- 2*sorted_resids_cur[k_05]
    length_99 <- 2*sorted_resids_cur[k_01]
    
    cur_true <- cvae_pred_test[j,i]
    lwr_85 <- cur_true - sorted_resids_cur[k_15]
    upr_85 <- cur_true + sorted_resids_cur[k_15]
    lwr_90 <- cur_true - sorted_resids_cur[k_10]
    upr_90 <- cur_true + sorted_resids_cur[k_10]
    lwr_95 <- cur_true - sorted_resids_cur[k_05]
    upr_95 <- cur_true + sorted_resids_cur[k_05]
    lwr_99 <- cur_true - sorted_resids_cur[k_01]
    upr_99 <- cur_true + sorted_resids_cur[k_01]
    
    interval_score_85[j] <- length_85+(2/.15)*(lwr_85 - cur_true)*as.numeric(cur_true < lwr_85)+(2/.15)*(cur_true-upr_85)*as.numeric(cur_true > upr_85)
    interval_score_90[j] <- length_90+(2/.1)*(lwr_90 - cur_true)*as.numeric(cur_true < lwr_90)+(2/.1)*(cur_true -upr_90)*as.numeric(cur_true > upr_90)
    interval_score_95[j] <- length_95+(2/.05)*(lwr_95 - cur_true)*as.numeric(cur_true < lwr_95)+(2/.05)*(cur_true -upr_95)*as.numeric(cur_true > upr_95)
    interval_score_99[j] <- length_99+(2/.01)*(lwr_99 - cur_true)*as.numeric(cur_true < lwr_99)+(2/.01)*(cur_true -upr_99)*as.numeric(cur_true > upr_99)
  }
  print(i)
  coverage_cvae_conf[i,1] <- mean(coverage_15)
  coverage_cvae_conf[i,2] <- mean(coverage_10)
  coverage_cvae_conf[i,3] <- mean(coverage_05)
  coverage_cvae_conf[i,4] <- mean(coverage_01)
  length_cvae_conf[i,1] <- 2*sorted_resids_cur[k_15]
  length_cvae_conf[i,2] <- 2*sorted_resids_cur[k_10]
  length_cvae_conf[i,3] <- 2*sorted_resids_cur[k_05]
  length_cvae_conf[i,4] <- 2*sorted_resids_cur[k_01]
  int_scores_cvae_conf[i,1] <- mean(interval_score_85)
  int_scores_cvae_conf[i,2] <- mean(interval_score_90)
  int_scores_cvae_conf[i,3] <- mean(interval_score_95)
  int_scores_cvae_conf[i,4] <- mean(interval_score_99)
}

#C: kernel
coverage_kernel_conf <- matrix(rep(0,27*4), ncol = 4)
length_kernel_conf <- matrix(rep(0,27*4), ncol = 4)
int_scores_kernel_conf <- matrix(rep(0,27*4), ncol = 4)

for(i in 1:27){
  resids_cur <- abs(val.data.y.NEW[,i] - kernel_pred_val[,i])
  sorted_resids_cur <- sort(resids_cur)
  
  k_15 <- ceiling((119 + 1)*(1-.15))
  k_10 <- ceiling((119 + 1)*(1-.10))
  k_05 <- ceiling((119 + 1)*(1-.05))
  k_01 <- ceiling((119 + 1)*(1-.01))
  
  coverage_15 <- rep(0, 119)
  coverage_10 <- rep(0, 119)
  coverage_05 <- rep(0, 119)
  coverage_01 <- rep(0, 119)
  
  cur_test_sims <- test.data.y.NEW[,i]
  for(j in 1:119){
    if(((kernel_pred_test[j,i]+sorted_resids_cur[k_15]) >= cur_test_sims[j]) && ((kernel_pred_test[j,i]-sorted_resids_cur[k_15]) <= cur_test_sims[j])){
      coverage_15[j] <- 1
    }
    if(((kernel_pred_test[j,i]+sorted_resids_cur[k_10]) >= cur_test_sims[j]) && ((kernel_pred_test[j,i]-sorted_resids_cur[k_10]) <= cur_test_sims[j])){
      coverage_10[j] <- 1
    }
    if(((kernel_pred_test[j,i]+sorted_resids_cur[k_05]) >= cur_test_sims[j]) && ((kernel_pred_test[j,i]-sorted_resids_cur[k_05]) <= cur_test_sims[j])){
      coverage_05[j] <- 1
    }
    if(((kernel_pred_test[j,i]+sorted_resids_cur[k_01]) >= cur_test_sims[j]) && ((kernel_pred_test[j,i]-sorted_resids_cur[k_01]) <= cur_test_sims[j])){
      coverage_01[j] <- 1
    }
    length_85 <- 2*sorted_resids_cur[k_15]
    length_90 <- 2*sorted_resids_cur[k_10]
    length_95 <- 2*sorted_resids_cur[k_05]
    length_99 <- 2*sorted_resids_cur[k_01]
    
    cur_true <- kernel_pred_test[j,i]
    lwr_85 <- cur_true - sorted_resids_cur[k_15]
    upr_85 <- cur_true + sorted_resids_cur[k_15]
    lwr_90 <- cur_true - sorted_resids_cur[k_10]
    upr_90 <- cur_true + sorted_resids_cur[k_10]
    lwr_95 <- cur_true - sorted_resids_cur[k_05]
    upr_95 <- cur_true + sorted_resids_cur[k_05]
    lwr_99 <- cur_true - sorted_resids_cur[k_01]
    upr_99 <- cur_true + sorted_resids_cur[k_01]
    
    interval_score_85[j] <- length_85+(2/.15)*(lwr_85 - cur_true)*as.numeric(cur_true < lwr_85)+(2/.15)*(cur_true-upr_85)*as.numeric(cur_true > upr_85)
    interval_score_90[j] <- length_90+(2/.1)*(lwr_90 - cur_true)*as.numeric(cur_true < lwr_90)+(2/.1)*(cur_true -upr_90)*as.numeric(cur_true > upr_90)
    interval_score_95[j] <- length_95+(2/.05)*(lwr_95 - cur_true)*as.numeric(cur_true < lwr_95)+(2/.05)*(cur_true -upr_95)*as.numeric(cur_true > upr_95)
    interval_score_99[j] <- length_99+(2/.01)*(lwr_99 - cur_true)*as.numeric(cur_true < lwr_99)+(2/.01)*(cur_true -upr_99)*as.numeric(cur_true > upr_99)
  }
  print(i)
  coverage_kernel_conf[i,1] <- mean(coverage_15)
  coverage_kernel_conf[i,2] <- mean(coverage_10)
  coverage_kernel_conf[i,3] <- mean(coverage_05)
  coverage_kernel_conf[i,4] <- mean(coverage_01)
  length_kernel_conf[i,1] <- 2*sorted_resids_cur[k_15]
  length_kernel_conf[i,2] <- 2*sorted_resids_cur[k_10]
  length_kernel_conf[i,3] <- 2*sorted_resids_cur[k_05]
  length_kernel_conf[i,4] <- 2*sorted_resids_cur[k_01]
  int_scores_kernel_conf[i,1] <- mean(interval_score_85)
  int_scores_kernel_conf[i,2] <- mean(interval_score_90)
  int_scores_kernel_conf[i,3] <- mean(interval_score_95)
  int_scores_kernel_conf[i,4] <- mean(interval_score_99)
}

#coverage
name_vec <- c('kernel conf', 'kernel lm', 'CVAE conf', 'CVAE lm', 'NN conf', 'NN lm')
par(mfrow = c(2,2))
boxplot(coverage_kernel_conf[,1], coverages_kernel[,1], coverage_cvae_conf[,1], coverages_cvae[,1], coverage_nn_conf[,1], coverages_nn[,1], main = '85% coverage', names = name_vec, cex.axis=.85)
abline(h = .85, lty = 2)
boxplot(coverage_kernel_conf[,2], coverages_kernel[,2], coverage_cvae_conf[,2], coverages_cvae[,2], coverage_nn_conf[,2], coverages_nn[,2],main = '90% coverage', names = name_vec, cex.axis=.85)
abline(h = .90, lty = 2)
boxplot(coverage_kernel_conf[,3], coverages_kernel[,3], coverage_cvae_conf[,3], coverages_cvae[,3], coverage_nn_conf[,3], coverages_nn[,3],main = '95% coverage', names = name_vec, cex.axis=.85)
abline(h = .95, lty = 2)
boxplot(coverage_kernel_conf[,4], coverages_kernel[,4], coverage_cvae_conf[,4], coverages_cvae[,4], coverage_nn_conf[,4], coverages_nn[,4],main = '99% coverage', names = name_vec, cex.axis=.85)
abline(h = .99, lty = 2)
#length
par(mfrow = c(2,2))
boxplot(length_kernel_conf[,1], lengths_kernel[,1], length_cvae_conf[,1],lengths_cvae[,1], length_nn_conf[,1],lengths_nn[,1], main = '85% length', cex.axis=.85,names = name_vec)
boxplot(length_kernel_conf[,2], lengths_kernel[,2], length_cvae_conf[,2], lengths_cvae[,2], length_nn_conf[,1], lengths_nn[,2], main = '90% length', cex.axis=.85,names = name_vec)
boxplot(length_kernel_conf[,3], lengths_kernel[,3], length_cvae_conf[,3], lengths_cvae[,3], length_nn_conf[,1], lengths_nn[,3], main = '95% length', cex.axis=.85,names = name_vec)
boxplot(length_kernel_conf[,4], lengths_kernel[,4], length_cvae_conf[,4], lengths_cvae[,4], length_nn_conf[,1], lengths_nn[,4], main = '99% length', cex.axis=.85,names = name_vec)
#interval scores
par(mfrow = c(2,2))
boxplot(int_scores_kernel_conf[,1], int_scores_kernel[,1], int_scores_cvae_conf[,1],int_scores_cvae[,1], int_scores_nn_conf[,1],int_scores_nn[,1], main = '85% interval scores', cex.axis=1.5,cex.main = 2, names = name_vec)
boxplot(int_scores_kernel_conf[,2], int_scores_kernel[,2], int_scores_cvae_conf[,2],int_scores_cvae[,2], int_scores_nn_conf[,2],int_scores_nn[,2], main = '90% interval scores', cex.axis=1.5,cex.main = 2,names = name_vec)
boxplot(int_scores_kernel_conf[,3], int_scores_kernel[,3], int_scores_cvae_conf[,3],int_scores_cvae[,3], int_scores_nn_conf[,3],int_scores_nn[,3], main = '95% interval scores', cex.axis=1.5,cex.main = 2,names = name_vec)
boxplot(int_scores_kernel_conf[,4], int_scores_kernel[,4], int_scores_cvae_conf[,4],int_scores_cvae[,4], int_scores_nn_conf[,4],int_scores_nn[,4], main = '99% interval scores', cex.axis=1.5,cex.main = 2,names = name_vec)


#plot accuracies
# Load ggplot2 library
library(ggplot2)

#histogram for R^2
# Combine data into a data frame
data <- data.frame(
  value = c(accs_kernel, accs_cvae, accs_nn),
  group = factor(rep(c("Kernel", "CVAE", "NN"), each = 27))
)

# Create the boxplot
p <- ggplot(data, aes(x = group, y = value, fill = group)) +
  geom_boxplot(lwd = 1.1) +
  labs(title = "Accuracy of emulators",
       x = "Emulator",
       y = "R^2") +
  theme_minimal() +
  scale_fill_brewer(palette = "Pastel1") +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1)) 
p
#histogram for RMSE
#change to units of cm
rmse_kernel_cm <- 100*rmse_kernel
rmse_cvae_cm <- 100*rmse_cvae
rmse_nn_cm <- 100*rmse_nn
data2 <- data.frame(
  value = c(rmse_kernel_cm, rmse_cvae_cm, rmse_nn_cm),
  group = factor(rep(c("Kernel", "CVAE", "NN"), each = 27))
)

# Create the boxplot
q<- ggplot(data2, aes(x = group, y = value, fill = group)) +
  geom_boxplot(lwd = 1.1) +
  labs(title = "Accuracy of emulators",
       x = "Emulator",
       y = "RMSE (cm)") +
  theme_minimal() +
  scale_fill_brewer(palette = "Pastel1") +
  theme(text = element_text(size = 20),
        axis.text.x = element_text(angle = 90, hjust = 1))  
q

#paired Wilcoxon test for emulators based on RMSE and R^2
wilcox.test(rmse_kernel, rmse_cvae, alternative = "less", paired = TRUE)
wilcox.test(rmse_kernel, rmse_nn, alternative = "less", paired = TRUE)
wilcox.test(accs_kernel, accs_cvae, alternative = "greater", paired = TRUE)
wilcox.test(accs_kernel, accs_nn, alternative = "greater", paired = TRUE)

#Monte Carlo dropout coverages and lengths
summary(cvae_ensemble_coverage)
summary(cvae_ensemble_length*100) #cm