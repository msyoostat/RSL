#make sure to change working directory to where .RData files are
#e.g. setwd(...)
#make lm plots
load('Montevideo_lm_results.RData')
load('Midway_lm_results.RData')
load('Dunedin_lm_results.RData')

par(mfrow = c(3,3))

#Dunedin
#kernel
plot(Dunedin_kernel_pred, Dunedin_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Dunedin CVAE', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#cvae
plot(Dunedin_cvae_pred,Dunedin_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Dunedin CVAE', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#NN
plot(Dunedin_nn_pred,Dunedin_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Dunedin NN', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#Montevideo
#kernel
plot(Montevideo_kernel_pred,Montevideo_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Montevideo kernel',xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#cvae
plot(Montevideo_cvae_pred,Montevideo_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Montevideo CVAE', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#NN
plot(Montevideo_nn_pred,Montevideo_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Montevideo NN', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#Midway
#kernel
plot(Midway_kernel_pred,Midway_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Midway kernel', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#cvae
plot(Midway_cvae_pred,Midway_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Midway CVAE', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#NN
plot(Midway_nn_pred,Midway_sims, xlab = 'Predicted', ylab = 'Simulated', pch = 16, main = 'Midway NN', xlim = c(-.125, .35))
abline(a = 0, b  =1,lwd = 1.5)
#plot residual distributions
load('resids.RData')
par(mfrow = c(3,3))
#Dunedin
hist(abs(kernel_resids[,17]), main = 'Dunedin kernel', xlab = 'residual', xlim = c(0,.05))
hist(abs(CVAE_resids[,17]), main = 'Dunedin CVAE', xlab = 'residual', xlim = c(0,.05))
hist(abs(NN_resids[,17]), main = 'Dunedin NN', xlab = 'residual', xlim = c(0,.05))
#Montevideo
hist(abs(kernel_resids[,5]), main = 'Montevideo kernel', xlab = 'residual', xlim = c(0,.05))
hist(abs(CVAE_resids[,5]), main = 'Montevideo CVAE', xlab = 'residual', xlim = c(0,.05))
hist(abs(NN_resids[,5]), main = 'Montevideo NN', xlab = 'residual', xlim = c(0,.05))
#Midway
hist(abs(kernel_resids[,25]), main = 'Midway kernel', xlab = 'residual', xlim = c(0,.05))
hist(abs(CVAE_resids[,25]), main = 'Midway CVAE', xlab = 'residual', xlim = c(0,.05))
hist(abs(NN_resids[,25]), main = 'Midway NN', xlab = 'residual', xlim = c(0,.05))
