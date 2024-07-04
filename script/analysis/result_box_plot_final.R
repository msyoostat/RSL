library(ggplot2)
library("patchwork")

## 1200 * 1000

result<-read.csv("result_table.csv",header=TRUE)
result_reduced<-read.csv("result_reduced_table.csv",header=TRUE)

colnames(result)
colnames(result_reduced)

length_idx<- which(grepl("length",colnames(result))==TRUE )
coverage_idx<- which(grepl("coverage",colnames(result))==TRUE )
rsquared_idx<- which(grepl("rsquared",colnames(result))==TRUE )

length_reduced_idx<- which(grepl("length",colnames(result_reduced))==TRUE )
coverage_reduced_idx<- which(grepl("coverage",colnames(result_reduced))==TRUE )
rsquared_reduced_idx<- which(grepl("rsquared",colnames(result_reduced))==TRUE )
length(rsquared_reduced_idx)+length(coverage_reduced_idx)+length(length_reduced_idx)==ncol(result_reduced)

result_length<- result[,length_idx]
result_coverage<- result[,coverage_idx]
result_rsquared<- result[,rsquared_idx]

result_reduced_length<- result_reduced[,length_reduced_idx]
result_reduced_coverage<- result_reduced[,coverage_reduced_idx]
result_reduced_rsquared<- result_reduced[,rsquared_reduced_idx]

# coverage, length, rsquared ----------------------------------------------------------------
colnames(result_reduced_length)

cvae_length<- result_length[,1:4]
cvae_coverage<- result_coverage[,1:4]
cvae_r_squared <-result_rsquared[,1]

cvae_reduced_length<- result_reduced_length[,1:4]
cvae_reduced_coverage<- result_reduced_coverage[,1:4]
cvae_reduced_r_squared <-result_reduced_rsquared[,1]

fnn_length<- result_length[,5:8]
fnn_coverage<- result_coverage[,5:8]
fnn_r_squared <-result_rsquared[,2]

fnn_reduced_length<- result_reduced_length[,5:8]
fnn_reduced_coverage<- result_reduced_coverage[,5:8]
fnn_reduced_r_squared <-result_reduced_rsquared[,2]

rf_length <- result_length[,9:12]
rf_coverage<- result_coverage[,9:12]
rf_r_squared <-result_rsquared[,3]

rf_reduced_length <- result_reduced_length[,9:12]
rf_reduced_coverage<- result_reduced_coverage[,9:12]
rf_reduced_r_squared <-result_reduced_rsquared[,3]

gp_length <-result_length[,13:16]
gp_coverage<- result_coverage[,13:16]
gp_r_squared <-result_rsquared[,4]

gp_reduced_length <-result_reduced_length[,13:16]
gp_reduced_coverage<- result_reduced_coverage[,13:16]
gp_reduced_r_squared <-result_reduced_rsquared[,4]


## 99% coverage rate
#mean(cvae_99[,1])
#mean(fnn_99[,1])
cvae_99<-data.frame(coverage_rate=cvae_coverage$cvae_99_coverage,model=rep("CVAE",27))
fnn_99<-data.frame(coverage_rate=fnn_coverage$fnn_99_coverage,model=rep("NN",27))
rf_99<-data.frame(coverage_rate=rf_coverage$rf_99_coverage,model=rep("RF",27))
gp_99<-data.frame(coverage_rate=gp_coverage$gp_99_coverage,model=rep("GP",27))

cvae_reduced_99<-data.frame(coverage_rate=cvae_reduced_coverage$cvae_reduced_99_coverage,model=rep("CVAE_r",27))
fnn_reduced_99<-data.frame(coverage_rate=fnn_reduced_coverage$fnn_reduced_99_coverage,model=rep("NN_r",27))
rf_reduced_99<-data.frame(coverage_rate=rf_reduced_coverage$rf_reduced_99_coverage,model=rep("RF_r",27))
gp_reduced_99<-data.frame(coverage_rate=gp_reduced_coverage$gp_reduced_99_coverage,model=rep("GP_r",27))

#all_99<-rbind.data.frame(cvae_99, cvae_reduced_99, fnn_99, fnn_reduced_99, rf_99, rf_reduced_99, gp_99,gp_reduced_99)
all_99<-rbind.data.frame(cvae_99, fnn_99,rf_99, gp_99)
#all_99<-rbind.data.frame(cvae_reduced_99, fnn_reduced_99, rf_reduced_99, gp_reduced_99)


desired_order<-c("NN","CVAE","GP","RF")
all_99$model<-factor(all_99$model,levels=desired_order)
fig1<-ggplot(all_99, aes(x=model, y=coverage_rate,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("99% coverage rate")+
  geom_hline(yintercept=0.99,col="red",linetype="dashed")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
        )+
  xlab(NULL)+
  ylab("coverage rate")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

cvae_95<-data.frame(coverage_rate=cvae_coverage$cvae_95_coverage,model=rep("CVAE",27))
fnn_95<-data.frame(coverage_rate=fnn_coverage$fnn_95_coverage,model=rep("NN",27))
rf_95<-data.frame(coverage_rate=rf_coverage$rf_95_coverage,model=rep("RF",27))
gp_95<-data.frame(coverage_rate=gp_coverage$gp_95_coverage,model=rep("GP",27))

cvae_reduced_95<-data.frame(coverage_rate=cvae_reduced_coverage$cvae_reduced_95_coverage,model=rep("CVAE_r",27))
fnn_reduced_95<-data.frame(coverage_rate=fnn_reduced_coverage$fnn_reduced_95_coverage,model=rep("NN_r",27))
rf_reduced_95<-data.frame(coverage_rate=rf_reduced_coverage$rf_reduced_95_coverage,model=rep("RF_r",27))
gp_reduced_95<-data.frame(coverage_rate=gp_reduced_coverage$gp_reduced_95_coverage,model=rep("GP_r",27))

round(mean(cvae_reduced_95[,1]),3)
round(mean(fnn_reduced_95[,1]),3)
round(mean(rf_reduced_95[,1]),3)
round(mean(gp_reduced_95[,1]),3)

#all_95<-rbind.data.frame(cvae_95, cvae_reduced_95, fnn_95, fnn_reduced_95, rf_95, rf_reduced_95, gp_95,gp_reduced_95)
all_95<-rbind.data.frame(cvae_95, fnn_95, rf_95, gp_95)
#all_95<-rbind.data.frame(cvae_reduced_95, fnn_reduced_95, rf_reduced_95, gp_reduced_95)

desired_order<-c("NN","CVAE","GP","RF")
all_95$model<-factor(all_95$model,levels=desired_order)

fig2<-ggplot(all_95, aes(x=model, y=coverage_rate,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("95% coverage rate")+
  geom_hline(yintercept=0.95,col="red",linetype="dashed")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylab("coverage rate")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

cvae_90<-data.frame(coverage_rate=cvae_coverage$cvae_90_coverage,model=rep("CVAE",27))
fnn_90<-data.frame(coverage_rate=fnn_coverage$fnn_90_coverage,model=rep("NN",27))
rf_90<-data.frame(coverage_rate=rf_coverage$rf_90_coverage,model=rep("RF",27))
gp_90<-data.frame(coverage_rate=gp_coverage$gp_90_coverage,model=rep("GP",27))

cvae_reduced_90<-data.frame(coverage_rate=cvae_reduced_coverage$cvae_reduced_90_coverage,model=rep("CVAE_r",27))
fnn_reduced_90<-data.frame(coverage_rate=fnn_reduced_coverage$fnn_reduced_90_coverage,model=rep("NN_r",27))
rf_reduced_90<-data.frame(coverage_rate=rf_reduced_coverage$rf_reduced_90_coverage,model=rep("RF_r",27))
gp_reduced_90<-data.frame(coverage_rate=gp_reduced_coverage$gp_reduced_90_coverage,model=rep("GP_r",27))

#all_90<-rbind.data.frame(cvae_90, cvae_reduced_90, fnn_90, fnn_reduced_90, rf_90, rf_reduced_90, gp_90,gp_reduced_90)
all_90<-rbind.data.frame(cvae_90,  fnn_90, rf_90,  gp_90)
#all_90<-rbind.data.frame(cvae_reduced_90, fnn_reduced_90, rf_reduced_90, gp_reduced_90)
desired_order<-c("NN","CVAE","GP","RF")
all_90$model<-factor(all_90$model,levels=desired_order)


fig3<-ggplot(all_90, aes(x=model, y=coverage_rate,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("90% coverage rate")+
  geom_hline(yintercept=0.90,col="red",linetype="dashed")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylab("coverage rate")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))


cvae_85<-data.frame(coverage_rate=cvae_coverage$cvae_85_coverage,model=rep("CVAE",27))
fnn_85<-data.frame(coverage_rate=fnn_coverage$fnn_85_coverage,model=rep("NN",27))
rf_85<-data.frame(coverage_rate=rf_coverage$rf_85_coverage,model=rep("RF",27))
gp_85<-data.frame(coverage_rate=gp_coverage$gp_85_coverage,model=rep("GP",27))

cvae_reduced_85<-data.frame(coverage_rate=cvae_reduced_coverage$cvae_reduced_85_coverage,model=rep("CVAE_r",27))
fnn_reduced_85<-data.frame(coverage_rate=fnn_reduced_coverage$fnn_reduced_85_coverage,model=rep("NN_r",27))
rf_reduced_85<-data.frame(coverage_rate=rf_reduced_coverage$rf_reduced_85_coverage,model=rep("RF_r",27))
gp_reduced_85<-data.frame(coverage_rate=gp_reduced_coverage$gp_reduced_85_coverage,model=rep("GP_r",27))

#all_85<-rbind.data.frame(cvae_85, cvae_reduced_85, fnn_85, fnn_reduced_85, rf_85, rf_reduced_85, gp_85,gp_reduced_85)
all_85<-rbind.data.frame(cvae_85, fnn_85, rf_85,  gp_85)
#all_85<-rbind.data.frame( cvae_reduced_85,fnn_reduced_85, rf_reduced_85, gp_reduced_85)
desired_order<-c("NN","CVAE","GP","RF")
all_85$model<-factor(all_85$model,levels=desired_order)


fig4<-ggplot(all_85, aes(x=model, y=coverage_rate,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("85% coverage rate")+
  geom_hline(yintercept=0.85,col="red",linetype="dashed")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylab("coverage rate")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

combined<-fig1+fig2+fig3+fig4 &theme(legend.position="right")
combined+plot_layout(guides="collect")

### length

cvae_99<-data.frame(length=cvae_length$cvae_99_length,model=rep("CVAE",27))
fnn_99<-data.frame(length=fnn_length$fnn_99_length,model=rep("NN",27))
rf_99<-data.frame(length=rf_length$rf_99_length,model=rep("RF",27))
gp_99<-data.frame(length=gp_length$gp_99_length,model=rep("GP",27))

cvae_reduced_99<-data.frame(length=cvae_reduced_length$cvae_reduced_99_length,model=rep("CVAE_r",27))
fnn_reduced_99<-data.frame(length=fnn_reduced_length$fnn_reduced_99_length,model=rep("NN_r",27))
rf_reduced_99<-data.frame(length=rf_reduced_length$rf_reduced_99_length,model=rep("RF_r",27))
gp_reduced_99<-data.frame(length=gp_reduced_length$gp_reduced_99_length,model=rep("GP_r",27))

#all_99<-rbind.data.frame(cvae_99, cvae_reduced_99, fnn_99, fnn_reduced_99, rf_99, rf_reduced_99, gp_99,gp_reduced_99)
all_99<-rbind.data.frame(cvae_99, fnn_99,  rf_99,  gp_99)
#all_99<-rbind.data.frame( cvae_reduced_99,  fnn_reduced_99,  rf_reduced_99,gp_reduced_99)

desired_order<-c("NN","CVAE","GP","RF")
all_99$model<-factor(all_99$model,levels=desired_order)

fig11<-ggplot(all_99, aes(x=model, y=length,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("99% length")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylim(0,0.1)+
  ylab("length")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

cvae_95<-data.frame(length=cvae_length$cvae_95_length,model=rep("CVAE",27))
fnn_95<-data.frame(length=fnn_length$fnn_95_length,model=rep("NN",27))
rf_95<-data.frame(length=rf_length$rf_95_length,model=rep("RF",27))
gp_95<-data.frame(length=gp_length$gp_95_length,model=rep("GP",27))

cvae_reduced_95<-data.frame(length=cvae_reduced_length$cvae_reduced_95_length,model=rep("CVAE_r",27))
fnn_reduced_95<-data.frame(length=fnn_reduced_length$fnn_reduced_95_length,model=rep("NN_r",27))
rf_reduced_95<-data.frame(length=rf_reduced_length$rf_reduced_95_length,model=rep("RF_r",27))
gp_reduced_95<-data.frame(length=gp_reduced_length$gp_reduced_95_length,model=rep("GP_r",27))

#all_95<-rbind.data.frame(cvae_95, cvae_reduced_95, fnn_95, fnn_reduced_95, rf_95, rf_reduced_95, gp_95,gp_reduced_95)
all_95<-rbind.data.frame(cvae_95, fnn_95,  rf_95, gp_95)
#all_95<-rbind.data.frame( cvae_reduced_95, fnn_reduced_95,  rf_reduced_95, gp_reduced_95)
desired_order<-c("NN","CVAE","GP","RF")
all_95$model<-factor(all_95$model,levels=desired_order)


round(mean(cvae_reduced_95[,1]),3)
round(mean(fnn_reduced_95[,1]),3)
round(mean(rf_reduced_95[,1]),3)
round(mean(gp_reduced_95[,1]),3)

fig22<-ggplot(all_95, aes(x=model, y=length,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("95% length")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylim(0,0.1)+
  ylab("length")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

cvae_90<-data.frame(length=cvae_length$cvae_90_length,model=rep("CVAE",27))
fnn_90<-data.frame(length=fnn_length$fnn_90_length,model=rep("NN",27))
rf_90<-data.frame(length=rf_length$rf_90_length,model=rep("RF",27))
gp_90<-data.frame(length=gp_length$gp_90_length,model=rep("GP",27))

cvae_reduced_90<-data.frame(length=cvae_reduced_length$cvae_reduced_90_length,model=rep("CVAE_r",27))
fnn_reduced_90<-data.frame(length=fnn_reduced_length$fnn_reduced_90_length,model=rep("NN_r",27))
rf_reduced_90<-data.frame(length=rf_reduced_length$rf_reduced_90_length,model=rep("RF_r",27))
gp_reduced_90<-data.frame(length=gp_reduced_length$gp_reduced_90_length,model=rep("GP_r",27))

#all_90<-rbind.data.frame(cvae_90, cvae_reduced_90, fnn_90, fnn_reduced_90, rf_90, rf_reduced_90, gp_90,gp_reduced_90)
all_90<-rbind.data.frame(cvae_90,  fnn_90, rf_90,  gp_90)
#all_90<-rbind.data.frame(cvae_reduced_90,fnn_reduced_90,rf_reduced_90,gp_reduced_90)
desired_order<-c("NN","CVAE","GP","RF")
all_90$model<-factor(all_90$model,levels=desired_order)

fig33<-ggplot(all_90, aes(x=model, y=length,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("90% length")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylim(0,0.1)+
  ylab("length")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))


cvae_85<-data.frame(length=cvae_length$cvae_85_length,model=rep("CVAE",27))
fnn_85<-data.frame(length=fnn_length$fnn_85_length,model=rep("NN",27))
rf_85<-data.frame(length=rf_length$rf_85_length,model=rep("RF",27))
gp_85<-data.frame(length=gp_length$gp_85_length,model=rep("GP",27))

cvae_reduced_85<-data.frame(length=cvae_reduced_length$cvae_reduced_85_length,model=rep("CVAE_r",27))
fnn_reduced_85<-data.frame(length=fnn_reduced_length$fnn_reduced_85_length,model=rep("NN_r",27))
rf_reduced_85<-data.frame(length=rf_reduced_length$rf_reduced_85_length,model=rep("RF_r",27))
gp_reduced_85<-data.frame(length=gp_reduced_length$gp_reduced_85_length,model=rep("GP_r",27))

#all_85<-rbind.data.frame(cvae_85, cvae_reduced_85, fnn_85, fnn_reduced_85, rf_85, rf_reduced_85, gp_85,gp_reduced_85)
all_85<-rbind.data.frame(cvae_85,  fnn_85,  rf_85, gp_85)
#all_85<-rbind.data.frame(cvae_reduced_85, fnn_reduced_85,  rf_reduced_85,gp_reduced_85)
desired_order<-c("NN","CVAE","GP","RF")
all_85$model<-factor(all_85$model,levels=desired_order)

fig44<-ggplot(all_85, aes(x=model, y=length,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("85% length")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylim(0,0.1)+
  ylab("length")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

combined<-fig11+fig22+fig33+fig44 &theme(legend.position="right")
combined+plot_layout(guides="collect")

## r squared
cvae_r_squared<-data.frame(r_squared=cvae_r_squared,model=rep("CVAE",27))
fnn_r_squared<-data.frame(r_squared=fnn_r_squared,model=rep("NN",27))
rf_r_squared<-data.frame(r_squared=rf_r_squared,model=rep("RF",27))
gp_r_squared<-data.frame(r_squared=gp_r_squared,model=rep("GP",27))

round(mean(cvae_r_squared[,1]),3)
round(mean(fnn_r_squared[,1]),3)

round(mean(rf_r_squared[,1]),3)
round(mean(gp_r_squared[,1]),3)

cvae_reduced_r_squared<-data.frame(r_squared=cvae_reduced_r_squared,model=rep("CVAE_r",27))
fnn_reduced_r_squared<-data.frame(r_squared=fnn_reduced_r_squared,model=rep("NN_r",27))
rf_reduced_r_squared<-data.frame(r_squared=rf_reduced_r_squared,model=rep("RF_r",27))
gp_reduced_r_squared<-data.frame(r_squared=gp_reduced_r_squared,model=rep("GP_r",27))

#all_r_squared<-rbind.data.frame(cvae_r_squared, cvae_reduced_r_squared, fnn_r_squared, fnn_reduced_r_squared,
#                                rf_r_squared,rf_reduced_r_squared, gp_r_squared,gp_reduced_r_squared)

all_r_squared<-rbind.data.frame(cvae_r_squared, fnn_r_squared,
                                 rf_r_squared, gp_r_squared)
desired_order<-c("NN","CVAE","GP","RF")
all_r_squared$model<-factor(all_r_squared$model,levels=desired_order)


ggplot(all_r_squared, aes(x=model, y=r_squared,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("R-squared")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylab("R-squared")+
  scale_x_discrete(limits=c("NN","CVAE","GP","RF"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

rm(all_r_squared)
## r squared (reduced)
all_r_squared<-rbind.data.frame(cvae_reduced_r_squared, fnn_reduced_r_squared,
                               rf_reduced_r_squared, gp_reduced_r_squared)

desired_order<-c("NN_r","CVAE_r","GP_r","RF_r")
all_r_squared$model<-factor(all_r_squared$model,levels=desired_order)


ggplot(all_r_squared, aes(x=model, y=r_squared,fill=model)) + 
  geom_boxplot()+
  stat_summary(fun=mean, geom="point", shape=8,color="black", fill="red",size=3) +
  ggtitle("R-squared")+
  theme(plot.title = element_text(size=22,hjust = 0.5),axis.text=element_text(size=20,face="bold"),
        axis.title=element_text(size=20,face="bold"),
        legend.title=element_text(size=25,face="bold"),
        legend.text=element_text(size=25,face="bold"),
        axis.text.y = element_text(face="bold",size=20),
        axis.text.x = element_text(face="bold",size=20)
  )+
  xlab(NULL)+
  ylab("R-squared")+
  scale_x_discrete(limits=c("NN_r","CVAE_r","GP_r","RF_r"))+
  theme(aspect.ratio=1,axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))

