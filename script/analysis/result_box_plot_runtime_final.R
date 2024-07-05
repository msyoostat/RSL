library(ggplot2)
library("patchwork")
result<-read.csv("result_runtime.csv",header=TRUE)

colnames(result)
result<-result[,c(1:7)]
colnames(result)

# runtime ----------------------------------------------------------------


## runtime
#mean(cvae_99[,1])
#mean(fnn_99[,1])
cvae<-data.frame(run_time=result$cvae,model=rep("CVAE",nrow(result)))
fnn<-data.frame(run_time=result$fnn,model=rep("NN",nrow(result)))
rf<-data.frame(run_time=result$rf,model=rep("RF",nrow(result)))
gp<-data.frame(run_time=result$gp,model=rep("GP",nrow(result)))

cvae_cali<-data.frame(run_time=result$cvae_cali,model=rep("CVAE_cali",nrow(result)))
fnn_cali<-data.frame(run_time=result$fnn_cali,model=rep("NN_cali",nrow(result)))
rf_cali<-data.frame(run_time=result$rf_cali,model=rep("RF_cali",nrow(result)))


#all_99<-rbind.data.frame(cvae_99, cvae_reduced_99, fnn_99, fnn_reduced_99, rf_99, rf_reduced_99, gp_99,gp_reduced_99)
all<-rbind.data.frame(cvae, fnn,rf, gp)
desired_order<-c("NN","CVAE","GP","RF")
all$model<-factor(all$model,levels=desired_order)

#all_99<-rbind.data.frame(cvae_reduced_99, fnn_reduced_99, rf_reduced_99, gp_reduced_99)

ggplot(all, aes(x=model, y=run_time,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("Runtime in seconds")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylab("seconds")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

all_cali<-rbind.data.frame(cvae, cvae_cali, fnn, fnn_cali, rf, rf_cali, gp)

#all_99<-rbind.data.frame(cvae_reduced_99, fnn_reduced_99, rf_reduced_99, gp_reduced_99)

ggplot(all_cali, aes(x=model, y=run_time,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("Runtime in seconds")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"),
        legend.title=element_text(size=14,face="bold"),
        legend.text=element_text(size=14,face="bold"),
        axis.text.y = element_text(face="bold"),
        axis.text.x = element_text(face="bold")
  )+
  ylab("seconds")+
  scale_x_discrete(limits=c("NN","NN_cali","cvae","cvae_cali","gp","rf","rf_cali"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

mean( mean( rf_cali[,1]-rf[,1] ), 
      mean( fnn_cali[,1]-fnn[,1] ),  
      mean( (cvae_cali[,1]-cvae[,1])^2 )  )

