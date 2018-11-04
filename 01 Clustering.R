library(EMCluster)

abalone_full <- read.csv("abalone.csv", header=T)
abalone_full$binaryClass <- ifelse(abalone_full$binaryClass == 'N', 0, 1)
abalone <- abalone_full[-9]
Gender <- ifelse(abalone$Sex == 'M', 1, 
                 ifelse(abalone$Sex == 'F', 0, 0.5))
abalone <- data.frame(as.data.frame(Gender), abalone[-1])

churn_full <- read.csv("churn.csv", header=T)
churn <- churn_full[-21]

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
optimal_k(abalone, 25, 100)
classification_error(abalone, abalone_full$binaryClass, "Hartigan-Wong", 6)
classification_error(abalone, abalone_full$binaryClass, "Hartigan-Wong", 2)

set.seed(100)
optimal_k(churn, 15, 10)
classification_error(churn, churn_full$class, "Hartigan-Wong", 4)
classification_error(churn, churn_full$class, "Hartigan-Wong", 2)

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
optimal_centers(abalone, 6, 2)
classification_error(abalone, abalone_full$binaryClass, 5)
classification_error(abalone, abalone_full$binaryClass, 2)

set.seed(100)
optimal_centers(churn, 6, 2)
classification_error(churn, churn_full$class, 5)
classification_error(churn, churn_full$class, 2)

