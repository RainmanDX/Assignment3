library(ica)
library(RandPro)

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
screeplot(pca_model_abalone, type="lines")
print(summary(pca_model_abalone))
print(pca_model_abalone$rotation)
#abalone_pca <- as.data.frame(pca_model_abalone$x[,1:3])

pca_model_churn <- prcomp(churn_scaled)
screeplot(pca_model_churn, type="lines")
print(summary(pca_model_churn))
print(pca_model_churn$rotation)
#churn_pca <- as.data.frame(pca_model_churn$x[,1:15])

#################### INDEPENDENT COMPONENT ANALYSIS ####################

ica_model_abalone <- icafast(abalone_scaled, nc=8)
print(ica_model_abalone$R)
print(ica_model_abalone$vafs)
total_variance <- c()
for (i in 1:length(ica_model_abalone$vafs)) {
  total_variance[i] <- sum(ica_model_abalone$vafs[1:i])
}
print(total_variance)
plot(total_variance, xlab="Independent Components", 
     ylab="Proportion of Variance Explained"); lines(total_variance)
#abalone_ica <- ica_model_abalone$S[,1:2]

ica_model_churn <- icafast(churn_scaled, nc=20)
print(ica_model_churn$R)
print(ica_model_churn$vafs)
total_variance <- c()
for (i in 1:length(ica_model_churn$vafs)) {
  total_variance[i] <- sum(ica_model_churn$vafs[1:i])
}
print(total_variance)
plot(total_variance, xlab="Independent Components", 
     ylab="Proportion of Variance Explained"); lines(total_variance)
#churn_ica <- ica_model_churn$S[,1:15]

################### RANDOMIZED PROJECTION #######################

set.seed(100)
maximal_error <- 0.1
randomatrix <- form_matrix(2, 8, FALSE, eps=maximal_error)
print(dim(randomatrix))
print(dim(abalone_scaled))
abalone_rca <- as.data.frame(as.matrix(abalone_scaled) %*% t(randomatrix))
print(dim(abalone_rca))

set.seed(100)
maximal_error <- 0.1
randomatrix <- form_matrix(10, 20, FALSE, eps=maximal_error)
print(dim(randomatrix))
print(dim(churn_scaled))
churn_rca <- as.data.frame(as.matrix(churn_scaled) %*% t(randomatrix))
print(dim(churn_rca))

########## MODIFIED FORWARD SELECTION W/ REGRESSION #############

# Abalone Dataset
set.seed(100)

## Step 1
regression_model_8 <- lm(binaryClass~., data=abalone_full)
print(summary(regression_model_8))  
### Add Shucked.weight (#6 @ '< 2e-16')
stepwise_model_6 <- lm(binaryClass~., data=abalone_full[, c("Shucked.weight", 
                                                            "binaryClass")])
print(summary(stepwise_model_6))

## Step 2.i
regression_model_7 <- lm(binaryClass~., data=abalone_full[,-c(6)])
print(summary(regression_model_7))  
### Add Shell.weight (#8 @ '< 2e-16')
stepwise_model_6_8 <- lm(binaryClass~., data=abalone_full[, c("Shucked.weight",
                                                              "Shell.weight",
                                                              "binaryClass")])
print(summary(stepwise_model_6_8))  
### None to Remove

## Step 2.ii
regression_model_6 <- lm(binaryClass~., data=abalone_full[,-c(6,8)])
print(summary(regression_model_6))  
### Add Diameter (#3 @ 8.15e-14)
stepwise_model_6_8_3 <- lm(binaryClass~., data=abalone_full[, c("Diameter",
                                                                "Shucked.weight",
                                                                "Shell.weight",
                                                                "binaryClass")])
print(summary(stepwise_model_6_8_3)) 
### None to Remove

## Step 2.iii
regression_model_5 <- lm(binaryClass~., data=abalone_full[,-c(6,8,3)])
print(summary(regression_model_5))  
### Add Height (#4 @ '< 2e-16')
stepwise_model_6_8_3_4 <- lm(binaryClass~., data=abalone_full[, c("Diameter",
                                                                "Height",
                                                                "Shucked.weight",
                                                                "Shell.weight",
                                                                "binaryClass")])
print(summary(stepwise_model_6_8_3_4)) 
### None to Remove

## Step 2.iv
regression_model_4 <- lm(binaryClass~., data=abalone_full[,-c(6,8,3,4)])
print(summary(regression_model_4))  
### Add Length (#2 @ 4.46e-07)
stepwise_model_6_8_3_4_2 <- lm(binaryClass~., data=abalone_full[, c("Length",
                                                                  "Diameter",
                                                                  "Height",
                                                                  "Shucked.weight",
                                                                  "Shell.weight",
                                                                  "binaryClass")])
print(summary(stepwise_model_6_8_3_4_2)) 
### None to Remove

## Step 2.v
regression_model_3 <- lm(binaryClass~., data=abalone_full[,-c(6,8,3,4,2)])
print(summary(regression_model_3))  
### Add Whole.weight (#5 @ '< 2e-16')
stepwise_model_6_8_3_4_2_5 <- lm(binaryClass~., data=abalone_full[, c("Length",
                                                                    "Diameter",
                                                                    "Height",
                                                                    "Whole.weight",
                                                                    "Shucked.weight",
                                                                    "Shell.weight",
                                                                    "binaryClass")])
print(summary(stepwise_model_6_8_3_4_2_5)) 
### Remove Length
stepwise_model_6_8_3_4_5 <- lm(binaryClass~., data=abalone_full[, c("Diameter",
                                                                  "Height",
                                                                  "Whole.weight",
                                                                  "Shucked.weight",
                                                                  "Shell.weight",
                                                                  "binaryClass")])
print(summary(stepwise_model_6_8_3_4_5))

## Step 2.vi
regression_model_2 <- lm(binaryClass~., data=abalone_full[,-c(6,8,3,4,2,5)])
print(summary(regression_model_2))  
### Add Viscera.weight (#7 @ '< 2e-16')
stepwise_model_6_8_3_4_5_7 <- lm(binaryClass~., data=abalone_full[, c("Diameter",
                                                                      "Height",
                                                                      "Whole.weight",
                                                                      "Shucked.weight",
                                                                      "Viscera.weight",
                                                                      "Shell.weight",
                                                                      "binaryClass")])
print(summary(stepwise_model_6_8_3_4_5_7)) 

## Step 2.vii
regression_model_1 <- lm(binaryClass~., data=abalone_full[,-c(6,8,3,4,2,5,7)])
print(summary(regression_model_1))  
### None to Add
print(summary(stepwise_model_6_8_3_4_5_7))
### Remove Viscera.weight
print(summary(stepwise_model_6_8_3_4_5))
### Step 2 Complete

# Churn Dataset
set.seed(100)

## Step 1
regression_model_20 <- lm(class~., data=churn_full)
print(summary(regression_model_20))
#### Add international_plan (#5 @ '< 2e-16')
stepwise_model_5 <- lm(class~., data=churn_full[, c("international_plan",
                                                    "class")])
print(summary(stepwise_model_5))

## Step 2.i
regression_model_19 <- lm(class~., data=churn_full[-c(5)])
print(summary(regression_model_19))
### Add number_customer_service_calls (#20 @ '< 2e-16)
stepwise_model_5_20 <- lm(class~., data=churn_full[, c("international_plan",
                                                       "number_customer_service_calls",
                                                       "class")])
print(summary(stepwise_model_5_20))

## Step 2.ii
regression_model_18 <- lm(class~., data=churn_full[-c(5,20)])
print(summary(regression_model_18))
### Add voice_mail_plan (#6 @ 1.07e-05)
stepwise_model_5_20_6 <- lm(class~., data=churn_full[, c("international_plan",
                                                       "voice_mail_plan",
                                                       "number_customer_service_calls",
                                                       "class")])
print(summary(stepwise_model_5_20_6))
### None to Remove

## Step 2.iii
regression_model_17 <- lm(class~., data=churn_full[-c(5,20,6)])
print(summary(regression_model_17))
### Add number_vmail_messages (#7 @ 1.71e-13)
stepwise_model_5_20_6_7 <- lm(class~., data=churn_full[, c("international_plan",
                                                         "voice_mail_plan",
                                                         "number_vmail_messages",
                                                         "number_customer_service_calls",
                                                         "class")])
print(summary(stepwise_model_5_20_6_7))
### None to Remove

## Step 2.iv
regression_model_16 <- lm(class~., data=churn_full[-c(5,20,6,7)])
print(summary(regression_model_16))
### Add total_intl_calls (#18 @ 0.000513)
stepwise_model_5_20_6_7_18 <- lm(class~., data=churn_full[, c("international_plan",
                                                           "voice_mail_plan",
                                                           "number_vmail_messages",
                                                           "total_intl_calls",
                                                           "number_customer_service_calls",
                                                           "class")])
print(summary(stepwise_model_5_20_6_7_18))
### Remove number_vmail_messages
stepwise_model_5_20_6_18 <- lm(class~., data=churn_full[, c("international_plan", 
                                                            "voice_mail_plan",
                                                            "total_intl_calls",
                                                            "number_customer_service_calls",
                                                            "class")])
print(summary(stepwise_model_5_20_6_18))

## Step 2.v
regression_model_15 <- lm(class~., data=churn_full[-c(5,20,6,7,18)])
print(summary(regression_model_15))
### None to Add
print(summary(stepwise_model_5_20_6_18))
### Step 2 Complete
