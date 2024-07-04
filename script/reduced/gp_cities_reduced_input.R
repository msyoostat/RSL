library(laGP)
library(plgp)
set.seed(1)
load("regional_cities_train_test_val_py.RData")

train_data_y <- as.matrix(train.data.y)
test_data_y <- as.matrix(test.data.y)
val_data_y <- as.matrix(val.data.y)
grid_subset_output_cities <- grid.subset.output.cities

rm(list=setdiff(ls(), c("train_data_y","test_data_y",
                        "val_data_y","grid_subset_output_cities","train_name","test_name","val_name",
                        "test_data_lithk_region_max_x","train_data_lithk_region_max_x","val_data_lithk_region_max_x",
                        "test_data_lithk_region_min_x","train_data_lithk_region_min_x","val_data_lithk_region_min_x",
                        "test_data_lithk_region_sum_x","train_data_lithk_region_sum_x","val_data_lithk_region_sum_x",
                        "test_data_lithk_region_mean_x","train_data_lithk_region_mean_x","val_data_lithk_region_mean_x")))

gc()

colnames(train_data_y)<-NULL
colnames(val_data_y)<-NULL
colnames(test_data_y)<-NULL


# read inputs from python ------------------------------------
train_data_lithk_region_mean_x<-read.table("train_data_lithk_region_mean_x.txt", header = FALSE, sep = ",", dec = ".")
test_data_lithk_region_mean_x<-read.table("test_data_lithk_region_mean_x.txt", header = FALSE, sep = ",", dec = ".")
val_data_lithk_region_mean_x<-read.table("val_data_lithk_region_mean_x.txt", header = FALSE, sep = ",", dec = ".")

train_data_lithk_region_mean_x<-as.matrix(train_data_lithk_region_mean_x)
test_data_lithk_region_mean_x<-as.matrix(test_data_lithk_region_mean_x)
val_data_lithk_region_mean_x<-as.matrix(val_data_lithk_region_mean_x)

colnames(train_data_lithk_region_mean_x)<-NULL
colnames(test_data_lithk_region_mean_x)<-NULL
colnames(val_data_lithk_region_mean_x)<-NULL

# scratch -----------------------------------------------------------------
## first start with this (covariance function: exp(-D/theta) + nugget)
eps <- sqrt(.Machine$double.eps)
D <- distance(train_data_lithk_region_mean_x)
DXX <- distance(test_data_lithk_region_mean_x)
DX <- distance(test_data_lithk_region_mean_x, train_data_lithk_region_mean_x)

nl <- function(par, D, Y) {
  theta <- par[1]                                       ## change 1
  g <- par[2]
  n <- length(Y)
  K <- exp(-D/(theta*10000)) + diag(g, n)                       ## change 2
  Ki <- solve(K)
  ldetK <- determinant(K, logarithm=TRUE)$modulus
  ll <- - (n/2)*log(t(Y) %*% Ki %*% Y) - (1/2)*ldetK
  counter <<- counter + 1
  return(-ll)
}

### do GP for each cities (27 cities)


prediction_mean <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
prediction_sd <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
parameter_mat <-matrix(NA, nrow=3,ncol=ncol(test_data_y))
rownames(parameter_mat)<-c("theta","g","tau_square")

for(j in 1: ncol(test_data_y)){
  counter <- 0
  out <- optim(c(0.1, 0.1*var(train_data_y[,j])), nl, method="L-BFGS-B", lower=eps, 
               upper=c(3000, var(train_data_y[,j])), D=D, Y=train_data_y[,j]) 
  parameter_mat[1:2,j]<-c(out$par)
  K <- exp(- D/(10000*out$par[1]) ) + diag(out$par[2], nrow(train_data_lithk_region_mean_x))
  Ki <- solve(K)
  #### tau square estimate
  tau2hat <- drop(t(train_data_y[,j]) %*% Ki %*% train_data_y[,j] / nrow(train_data_lithk_region_mean_x))
  parameter_mat[3,j]<-c(tau2hat)
  ### prediction
  KXX <- exp(-DXX/(10000*out$par[1])) + diag(out$par[2], ncol(DXX))
  KX <- exp(-DX/(10000*out$par[1]))
  mup <- KX %*% Ki %*% train_data_y[,j]
  Sigmap <- tau2hat*(KXX - KX %*% Ki %*% t(KX))
  sdp <- sqrt(diag(Sigmap))
  prediction_mean[,j]<-mup
  prediction_sd[,j]<-sdp
  print(paste0(j,"th location gp done"))
}

#length average
length_all_95 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
length_all_99 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
length_all_90 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
length_all_85 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
for(i in 1: nrow(test_data_y)){
  for( j in 1: ncol(test_data_y)){
    # 95%
    lower <-prediction_mean[i,j] + qnorm(0.025, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.975, 0, sd=prediction_sd[i,j])
    length_all_95[i,j]<-c(upper-lower)
    #99%
    lower <-prediction_mean[i,j] + qnorm(0.005, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.995, 0, sd=prediction_sd[i,j])
    length_all_99[i,j]<-c(upper-lower)
    #90%
    lower <-prediction_mean[i,j] + qnorm(0.05, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.95, 0, sd=prediction_sd[i,j])
    length_all_90[i,j]<-c(upper-lower)
    #85%
    lower <-prediction_mean[i,j] + qnorm(0.075, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.925, 0, sd=prediction_sd[i,j])
    length_all_85[i,j]<-c(upper-lower)
  }
}
apply(length_all_99,2,mean)
average_length<-cbind.data.frame(grid_subset_output_cities$city,apply(length_all_99,2,mean),
                 apply(length_all_95,2,mean),apply(length_all_90,2,mean),
                 apply(length_all_85,2,mean))
colnames(average_length)<-c("cities","99%","95%","90%","85%")
print("average length by cities")
average_length

#coverage rate
coverage_all_95 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
coverage_all_99 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
coverage_all_90 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
coverage_all_85 <-matrix(NA,nrow=nrow(test_data_y),ncol=ncol(test_data_y))
for(i in 1: nrow(test_data_y)){
  for( j in 1: ncol(test_data_y)){
    # 95%
    lower <-prediction_mean[i,j] + qnorm(0.025, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.975, 0, sd=prediction_sd[i,j])
    if(test_data_y[i,j]<= upper & test_data_y[i,j]>=lower ){
      coverage_all_95[i,j]<-1
    } else{
      coverage_all_95[i,j]<-0
    }
    # 99%
    lower <-prediction_mean[i,j] + qnorm(0.005, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.995, 0, sd=prediction_sd[i,j])
    if(test_data_y[i,j]<= upper & test_data_y[i,j]>=lower ){
      coverage_all_99[i,j]<-1
    } else{
      coverage_all_99[i,j]<-0
    }
    # 90%
    lower <-prediction_mean[i,j] + qnorm(0.05, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.95, 0, sd=prediction_sd[i,j])
    if(test_data_y[i,j]<= upper & test_data_y[i,j]>=lower ){
      coverage_all_90[i,j]<-1
    } else{
      coverage_all_90[i,j]<-0
    }
    # 85%
    lower <-prediction_mean[i,j] + qnorm(0.075, 0, sd=prediction_sd[i,j])
    upper <-prediction_mean[i,j] + qnorm(0.925, 0, sd=prediction_sd[i,j])
    if(test_data_y[i,j]<= upper & test_data_y[i,j]>=lower ){
      coverage_all_85[i,j]<-1
    } else{
      coverage_all_85[i,j]<-0
    }
  }
}

average_coverage<-cbind.data.frame(grid_subset_output_cities$city,apply(coverage_all_99,2,mean),
                                 apply(coverage_all_95,2,mean),apply(coverage_all_90,2,mean),
                                 apply(coverage_all_85,2,mean))
colnames(average_coverage)<-c("cities","99%","95%","90%","85%")
print("average coverage by cities")
average_coverage

## r squared
r_squared <-numeric(0)
j=1
for(j in 1: ncol(test_data_y)){
  lm_result<-lm(test_data_y[,j]~prediction_mean[,j])
  lm_result<-summary(lm_result)
  r_squared[j]<-lm_result$r.squared
}

r_squared_cities<-cbind.data.frame(grid_subset_output_cities$city,r_squared)
colnames(r_squared_cities)<-c("cities","r squared")
print("R squared by cities")
r_squared_cities

