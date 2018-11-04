library(neuralnet)

abalone <- read.csv("abalone.csv", header=T)
abalone$binaryClass <- ifelse(abalone$binaryClass == 'N', 0, 1)

maxima <- apply(abalone[,2:9], 2, max)
minima <- apply(abalone[,2:9], 2, min)
abalone_scaled <- as.data.frame(scale(abalone[,2:9], center = minima, scale = maxima - minima))
Gender <- ifelse(abalone$Sex == 'M', 1, 
                 ifelse(abalone$Sex == 'F', 0, 0.5))
abalone_scaled <- data.frame(as.data.frame(Gender), abalone_scaled)

########## MODIFIED FORWARD SELECTION W/ REGRESSION #############
abalone_mfs <- abalone_scaled[, c("Diameter",
                                  "Height",
                                  "Whole.weight",
                                  "Shucked.weight",
                                  "Shell.weight",
                                  "binaryClass")]

set.seed(100)
sample_index <- sample(1:nrow(abalone_scaled), 3340)
training_orig <- abalone_scaled[sample_index,]; row.names(training_orig) <- NULL
testing_orig <- abalone_scaled[-sample_index,]; row.names(testing_orig) <- NULL
training_mfs <- abalone_mfs[sample_index,]; row.names(training_mfs) <- NULL
testing_mfs <- abalone_mfs[-sample_index,]; row.names(testing_mfs) <- NULL

columns_orig <- names(training_orig)
formula_orig <- as.formula(paste("binaryClass ~", 
                                 paste(columns_orig[!columns_orig %in% "binaryClass"], 
                                       collapse = " + ")))
columns_mfs <- names(training_mfs)
formula_mfs <- as.formula(paste("binaryClass ~", 
                                paste(columns_mfs[!columns_mfs %in% "binaryClass"], 
                                      collapse = " + ")))

############### LEARNING CURVE ANALYSIS ################

### Original Dataset
sample_sizes <- c(10, 50, 100, 500, 1000, 2000, 3000, 3340)

sample_index_10   <- sample(1:nrow(training_orig), 10)
sample_index_50   <- sample(1:nrow(training_orig), 50)
sample_index_100  <- sample(1:nrow(training_orig), 100)
sample_index_500  <- sample(1:nrow(training_orig), 500)
sample_index_1000 <- sample(1:nrow(training_orig), 1000)
sample_index_2000 <- sample(1:nrow(training_orig), 2000)
sample_index_3000 <- sample(1:nrow(training_orig), 3000)

training_subset_10   <- training_orig[sample_index_10,]; row.names(training_subset_10) <- NULL
training_subset_50   <- training_orig[sample_index_50,]; row.names(training_subset_50) <- NULL
training_subset_100  <- training_orig[sample_index_100,]; row.names(training_subset_100) <- NULL
training_subset_500  <- training_orig[sample_index_500,]; row.names(training_subset_500) <- NULL
training_subset_1000 <- training_orig[sample_index_1000,]; row.names(training_subset_1000) <- NULL
training_subset_2000 <- training_orig[sample_index_2000,]; row.names(training_subset_2000) <- NULL
training_subset_3000 <- training_orig[sample_index_3000,]; row.names(training_subset_3000) <- NULL
training_subset_3340 <- training_orig

nnetwork10   <- neuralnet(formula_orig, data=training_subset_10, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork50   <- neuralnet(formula_orig, data=training_subset_50, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork100  <- neuralnet(formula_orig, data=training_subset_100, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork500  <- neuralnet(formula_orig, data=training_subset_500, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork1000 <- neuralnet(formula_orig, data=training_subset_1000, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork2000 <- neuralnet(formula_orig, data=training_subset_2000, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork3000 <- neuralnet(formula_orig, data=training_subset_3000, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork3340 <- neuralnet(formula_orig, data=training_subset_3340, hidden=3, linear.output=T, rep=2, stepmax=500000)

pred10.test   <- compute(nnetwork10, testing_orig[,1:8])
pred50.test   <- compute(nnetwork50, testing_orig[,1:8])
pred100.test  <- compute(nnetwork100, testing_orig[,1:8])
pred500.test  <- compute(nnetwork500, testing_orig[,1:8])
pred1000.test <- compute(nnetwork1000, testing_orig[,1:8])
pred2000.test <- compute(nnetwork2000, testing_orig[,1:8])
pred3000.test <- compute(nnetwork3000, testing_orig[,1:8])
pred3340.test <- compute(nnetwork3340, testing_orig[,1:8])

pred10.test   <- ifelse(pred10.test$net.result >= 0.5, 1, 0)
pred50.test   <- ifelse(pred50.test$net.result >= 0.5, 1, 0)
pred100.test  <- ifelse(pred100.test$net.result >= 0.5, 1, 0)
pred500.test  <- ifelse(pred500.test$net.result >= 0.5, 1, 0)
pred1000.test <- ifelse(pred1000.test$net.result >= 0.5, 1, 0)
pred2000.test <- ifelse(pred2000.test$net.result >= 0.5, 1, 0)
pred3000.test <- ifelse(pred3000.test$net.result >= 0.5, 1, 0)
pred3340.test <- ifelse(pred3340.test$net.result >= 0.5, 1, 0)

lc_testing_results <- c()
lc_testing_results[1] <- 1 - sum(pred10.test == testing_orig$binaryClass) / length(pred10.test)
lc_testing_results[2] <- 1 - sum(pred50.test == testing_orig$binaryClass) / length(pred50.test)
lc_testing_results[3] <- 1 - sum(pred100.test == testing_orig$binaryClass) / length(pred100.test)
lc_testing_results[4] <- 1 - sum(pred500.test == testing_orig$binaryClass) / length(pred500.test)
lc_testing_results[5] <- 1 - sum(pred1000.test == testing_orig$binaryClass) / length(pred1000.test)
lc_testing_results[6] <- 1 - sum(pred2000.test == testing_orig$binaryClass) / length(pred2000.test)
lc_testing_results[7] <- 1 - sum(pred3000.test == testing_orig$binaryClass) / length(pred3000.test)
lc_testing_results[8] <- 1 - sum(pred3340.test == testing_orig$binaryClass) / length(pred3340.test)

print(sample_sizes)
print(lc_testing_results)

plot(sample_sizes, lc_testing_results, xlab="Training Set Sample Size", ylab="Testing Set Prediction Error")
lines(sample_sizes, lc_testing_results)

### Modified Dataset

training_subset_10   <- training_mfs[sample_index_10,]; row.names(training_subset_10) <- NULL
training_subset_50   <- training_mfs[sample_index_50,]; row.names(training_subset_50) <- NULL
training_subset_100  <- training_mfs[sample_index_100,]; row.names(training_subset_100) <- NULL
training_subset_500  <- training_mfs[sample_index_500,]; row.names(training_subset_500) <- NULL
training_subset_1000 <- training_mfs[sample_index_1000,]; row.names(training_subset_1000) <- NULL
training_subset_2000 <- training_mfs[sample_index_2000,]; row.names(training_subset_2000) <- NULL
training_subset_3000 <- training_mfs[sample_index_3000,]; row.names(training_subset_3000) <- NULL
training_subset_3340 <- training_mfs

nnetwork10   <- neuralnet(formula_mfs, data=training_subset_10, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork50   <- neuralnet(formula_mfs, data=training_subset_50, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork100  <- neuralnet(formula_mfs, data=training_subset_100, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork500  <- neuralnet(formula_mfs, data=training_subset_500, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork1000 <- neuralnet(formula_mfs, data=training_subset_1000, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork2000 <- neuralnet(formula_mfs, data=training_subset_2000, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork3000 <- neuralnet(formula_mfs, data=training_subset_3000, hidden=3, linear.output=T, rep=2, stepmax=500000)
nnetwork3340 <- neuralnet(formula_mfs, data=training_subset_3340, hidden=3, linear.output=T, rep=2, stepmax=500000)

pred10.test   <- compute(nnetwork10, testing_mfs[,1:5])
pred50.test   <- compute(nnetwork50, testing_mfs[,1:5])
pred100.test  <- compute(nnetwork100, testing_mfs[,1:5])
pred500.test  <- compute(nnetwork500, testing_mfs[,1:5])
pred1000.test <- compute(nnetwork1000, testing_mfs[,1:5])
pred2000.test <- compute(nnetwork2000, testing_mfs[,1:5])
pred3000.test <- compute(nnetwork3000, testing_mfs[,1:5])
pred3340.test <- compute(nnetwork3340, testing_mfs[,1:5])

pred10.test   <- ifelse(pred10.test$net.result >= 0.5, 1, 0)
pred50.test   <- ifelse(pred50.test$net.result >= 0.5, 1, 0)
pred100.test  <- ifelse(pred100.test$net.result >= 0.5, 1, 0)
pred500.test  <- ifelse(pred500.test$net.result >= 0.5, 1, 0)
pred1000.test <- ifelse(pred1000.test$net.result >= 0.5, 1, 0)
pred2000.test <- ifelse(pred2000.test$net.result >= 0.5, 1, 0)
pred3000.test <- ifelse(pred3000.test$net.result >= 0.5, 1, 0)
pred3340.test <- ifelse(pred3340.test$net.result >= 0.5, 1, 0)

lc_testing_results <- c()
lc_testing_results[1] <- 1 - sum(pred10.test == testing_mfs$binaryClass) / length(pred10.test)
lc_testing_results[2] <- 1 - sum(pred50.test == testing_mfs$binaryClass) / length(pred50.test)
lc_testing_results[3] <- 1 - sum(pred100.test == testing_mfs$binaryClass) / length(pred100.test)
lc_testing_results[4] <- 1 - sum(pred500.test == testing_mfs$binaryClass) / length(pred500.test)
lc_testing_results[5] <- 1 - sum(pred1000.test == testing_mfs$binaryClass) / length(pred1000.test)
lc_testing_results[6] <- 1 - sum(pred2000.test == testing_mfs$binaryClass) / length(pred2000.test)
lc_testing_results[7] <- 1 - sum(pred3000.test == testing_mfs$binaryClass) / length(pred3000.test)
lc_testing_results[8] <- 1 - sum(pred3340.test == testing_mfs$binaryClass) / length(pred3340.test)

print(sample_sizes)
print(lc_testing_results)

plot(sample_sizes, lc_testing_results, xlab="Training Set Sample Size", ylab="Testing Set Prediction Error")
lines(sample_sizes, lc_testing_results)