library(EMCluster)
library(ica)
library(RandPro)
library(neuralnet)

abalone <- read.csv("abalone.csv", header=T)
abalone$binaryClass <- ifelse(abalone$binaryClass == 'N', 0, 1)

maxima <- apply(abalone[,2:9], 2, max)
minima <- apply(abalone[,2:9], 2, min)
abalone_scaled <- as.data.frame(scale(abalone[,2:9], center = minima, scale = maxima - minima))
Gender <- ifelse(abalone$Sex == 'M', 1, 
                 ifelse(abalone$Sex == 'F', 0, 0.5))
abalone_scaled <- data.frame(as.data.frame(Gender), abalone_scaled)

cluster_model <- init.EM(abalone[,-9], nclass=5)
Cluster <- cluster_model$class
abalone_scaled <- data.frame(abalone_scaled, as.data.frame(Cluster))
