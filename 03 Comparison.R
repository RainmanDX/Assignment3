library(ica)
library(RandPro)
library(EMCluster)

abalone_full <- read.csv("abalone.csv", header=T)
abalone_full$binaryClass <- ifelse(abalone_full$binaryClass == 'N', 0, 1)
Gender <- ifelse(abalone_full$Sex == 'M', 1, 
                 ifelse(abalone_full$Sex == 'F', 0, 0.5))
abalone_full <- data.frame(as.data.frame(Gender), abalone_full[,2:9])

maxima <- apply(abalone_full[,1:8], 2, max)
minima <- apply(abalone_full[,1:8], 2, min)
abalone_scaled <- as.data.frame(scale(abalone_full[,1:8], 
                                      center = minima, scale = maxima - minima))

churn_full <- read.csv("churn.csv", header=T)

maxima <- apply(churn_full[,1:20], 2, max)
minima <- apply(churn_full[,1:20], 2, min)
churn_scaled <- as.data.frame(scale(churn_full[,1:20], 
                                    center = minima, scale = maxima - minima))

#################### PRINCIPAL COMPONENT ANALYSIS ####################

pca_model_abalone <- prcomp(abalone_scaled)
abalone_pca <- as.data.frame(pca_model_abalone$x[,1:2])

pca_model_churn <- prcomp(churn_scaled)
churn_pca <- as.data.frame(pca_model_churn$x[,1:10])

#################### INDEPENDENT COMPONENT ANALYSIS ####################

ica_model_abalone <- icafast(abalone_scaled, nc=8)
abalone_ica <- as.data.frame(ica_model_abalone$S[,1:2])

ica_model_churn <- icafast(churn_scaled, nc=20)
churn_ica <- as.data.frame(ica_model_churn$S[,1:11])

################### RANDOMIZED PROJECTION #######################

set.seed(100)
maximal_error <- 0.1
randomatrix <- form_matrix(2, 8, FALSE, eps=maximal_error)
abalone_rca <- as.data.frame(as.matrix(abalone_scaled) %*% t(randomatrix))

set.seed(100)
maximal_error <- 0.1
randomatrix <- form_matrix(10, 20, FALSE, eps=maximal_error)
churn_rca <- as.data.frame(as.matrix(churn_scaled) %*% t(randomatrix))

########## MODIFIED FORWARD SELECTION W/ REGRESSION #############

abalone_mfs <- abalone_scaled[, c("Diameter",
                                  "Height",
                                  "Whole.weight",
                                  "Shucked.weight",
                                  "Shell.weight")]

churn_mfs <- churn_scaled[, c("international_plan",
                              "voice_mail_plan",
                              "total_intl_calls",
                              "number_customer_service_calls")]

################## k-MEANS CLUSTERING #######################

optimal_k <- function(dataset, max_k, tests) {
  k_range <- 2:max_k
  average_distances <- integer(max_k-1)
  for (k in k_range) {
    all_tests <- integer(tests)
    for (test_index in 1:tests) {
      model <- kmeans(dataset, centers=k, iter.max=100)
      all_tests[test_index] <- model$withinss
    }
    average_distances[k-1] <- mean(all_tests)
  }
  print(average_distances)
  plot(k_range, average_distances, type="b", 
       xlab="Cluster Centers", ylab="Total Squared Distance Within Clusters")
}

classification_error <- function(dataset, y_vector, clustering_algo, k) {
  model <- kmeans(dataset, centers=k, iter.max=100, algorithm=clustering_algo)
  cluster <- model$cluster
  prediction <- integer(length(cluster))
  for (i in 1:length(cluster)) {
    prediction[i] <- cluster[[i]]
  }
  accuracy_table <- table(prediction, y_vector)
  print(accuracy_table)
}

set.seed(100)
optimal_k(abalone_pca, 25, 100)
optimal_k(abalone_ica, 25, 100)
optimal_k(abalone_rca, 25, 100)
optimal_k(abalone_mfs, 25, 100)

classification_error(abalone_pca, abalone_full$binaryClass, "Hartigan-Wong", 6)
classification_error(abalone_ica, abalone_full$binaryClass, "Hartigan-Wong", 6)
classification_error(abalone_rca, abalone_full$binaryClass, "Hartigan-Wong", 6)
classification_error(abalone_mfs, abalone_full$binaryClass, "Hartigan-Wong", 6)

classification_error(abalone_pca, abalone_full$binaryClass, "Hartigan-Wong", 2)
classification_error(abalone_ica, abalone_full$binaryClass, "Hartigan-Wong", 2)
classification_error(abalone_rca, abalone_full$binaryClass, "Hartigan-Wong", 2)
classification_error(abalone_mfs, abalone_full$binaryClass, "Hartigan-Wong", 2)

set.seed(100)
optimal_k(churn_pca, 15, 10)
optimal_k(churn_ica, 15, 10)
optimal_k(churn_rca, 15, 10)
optimal_k(churn_mfs, 15, 10)

classification_error(churn_pca, churn_full$class, "Hartigan-Wong", 6)
classification_error(churn_ica, churn_full$class, "Hartigan-Wong", 6)
classification_error(churn_rca, churn_full$class, "Hartigan-Wong", 6)
classification_error(churn_mfs, churn_full$class, "Hartigan-Wong", 6)

classification_error(churn_pca, churn_full$class, "Hartigan-Wong", 2)
classification_error(churn_ica, churn_full$class, "Hartigan-Wong", 2)
classification_error(churn_rca, churn_full$class, "Hartigan-Wong", 2)
classification_error(churn_mfs, churn_full$class, "Hartigan-Wong", 2)

############## EXPECTATION MAXIMIZATION ##################

optimal_centers <- function(dataset, max_k, tests) {
  k_range <- 2:max_k
  average_distances <- integer(max_k-1)
  for (k in k_range) {
    all_tests <- integer(tests)
    for (test_index in 1:tests) {
      model <- init.EM(dataset, nclass=k)
      errors <- integer(length(model$class))
      for (i in 1:length(model$class)) {
        errors[i] <- sum((dataset[i,]-model$Mu[model$class[i],])^2)
      }
      all_tests[test_index] <- sum(errors)
    }
    average_distances[k-1] <- mean(all_tests)
  }
  print(average_distances)
  plot(k_range, average_distances, type="b",
       xlab="Cluster Centers", ylab="Total Squared Distance Within Clusters")
}

classification_error <- function(dataset, y_vector, k) {
  model <- init.EM(dataset, nclass=k)
  accuracy_table <- table(model$class, y_vector)
  print(accuracy_table)
}

set.seed(100)
optimal_centers(abalone_pca, 12, 2)
optimal_centers(abalone_ica, 12, 2)
optimal_centers(abalone_rca, 12, 2)
optimal_centers(abalone_mfs, 12, 2)

classification_error(abalone_pca, abalone_full$binaryClass, 5)
classification_error(abalone_ica, abalone_full$binaryClass, 5)
classification_error(abalone_rca, abalone_full$binaryClass, 5)
classification_error(abalone_mfs, abalone_full$binaryClass, 5)

classification_error(abalone_pca, abalone_full$binaryClass, 2)
classification_error(abalone_ica, abalone_full$binaryClass, 2)
classification_error(abalone_rca, abalone_full$binaryClass, 2)
classification_error(abalone_mfs, abalone_full$binaryClass, 2)

set.seed(100)
optimal_centers(churn_pca, 6, 2)
optimal_centers(churn_ica, 6, 2)
optimal_centers(churn_rca, 6, 2)
optimal_centers(churn_mfs, 4, 2)

classification_error(churn_pca, churn_full$class, 5)
classification_error(churn_ica, churn_full$class, 5)
classification_error(churn_rca, churn_full$class, 5)
classification_error(churn_mfs, churn_full$class, 5)

classification_error(churn_pca, churn_full$class, 2)
classification_error(churn_ica, churn_full$class, 2)
classification_error(churn_rca, churn_full$class, 2)
classification_error(churn_mfs, churn_full$class, 2)

