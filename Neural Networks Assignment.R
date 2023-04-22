### Load required libraries ###
library(keras)
library(tidyr)
library(dplyr)
library(magrittr)
library(Matrix)

### clean environment ###
rm(list = ls())

### load data ###
setwd("Path to your tensorflow environment")
NNdataOriginal <- read.csv("Dataset-part-2.csv")
NNdata <- NNdataOriginal

###Pre-Process Data ###

#Check structure of data
#NN work best without NA's and only with integers
#Check usefulness of the multiple class character variables.
str(NNdata)
#Inital nummeric values to normalize later:
#AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED

unique(NNdata$NAME_INCOME_TYPE)
#only 4 unique values -> Keep as factor
unique(NNdata$NAME_EDUCATION_TYPE)
#only 4 unique values -> Keep as factor
unique(NNdata$NAME_FAMILY_STATUS)
#only 5 unique values -> Keep as factor
unique(NNdata$NAME_HOUSING_TYPE)
#only 6 unique values -> Keep as factor
unique(NNdata$OCCUPATION_TYPE)
#18 values -> Keep as factor
#We do not check the variable status here because it's the target/dependent variable.

#Change all the character variables to factor except
NNdata$CODE_GENDER <- as.factor(NNdata$CODE_GENDER)
NNdata$FLAG_OWN_CAR <- as.factor(NNdata$FLAG_OWN_CAR)
NNdata$FLAG_OWN_REALTY <- as.factor(NNdata$FLAG_OWN_REALTY)
NNdata$NAME_INCOME_TYPE <- as.factor(NNdata$NAME_INCOME_TYPE)
NNdata$NAME_EDUCATION_TYPE <- as.factor(NNdata$NAME_EDUCATION_TYPE)
NNdata$NAME_FAMILY_STATUS <- as.factor(NNdata$NAME_FAMILY_STATUS)
NNdata$NAME_HOUSING_TYPE <- as.factor(NNdata$NAME_HOUSING_TYPE)
NNdata$OCCUPATION_TYPE <- as.factor(NNdata$OCCUPATION_TYPE)
NNdata$status <- as.factor(NNdata$status)
#See if it worked
str(NNdata)

#Change all factors and ints to nummeric
NNdata %<>% mutate_if(is.factor, as.numeric)
NNdata %<>% mutate_if(is.integer, as.numeric)
#See if it worked
str(NNdata)
#Check if int num converstion caused any errors.
sum(NNdata$ID - NNdataOriginal$ID)

#Check for na's
colSums(is.na(NNdata))
str(NNdata)

#Maybe we can also just delete the column occupation type... We could keep 20699 more observations this way. We will experiment later.
#NaOmitData <- select(NNdata, -c(OCCUPATION_TYPE))
NaOmitData <- na.omit(NNdata)
str(NaOmitData)
#ID is not useful
NaOmitData <- select(NaOmitData, -c(ID))
str(NaOmitData)
View(NaOmitData)
#Convert to Matrix -> Not necessary yet



#Create Training-, Dev-, and Test-Set
set.seed(1)
ind <- sample(3, nrow(NaOmitData), replace = T, prob = c(.7, .1, .2))
X_train <- NaOmitData[ind==1,1:17]
X_dev <- NaOmitData[ind==2, 1:17]
X_test <- NaOmitData[ind==3, 1:17]
Y_train <- NaOmitData[ind==1, 18]
Y_dev <- NaOmitData[ind==2, 18]
Y_test <- NaOmitData[ind==3, 18]


str(NaOmitData)
str(X_train)


#Normalize independent variables of X_train and X_dev for values: #AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED
m_inc_train <- mean(X_train$AMT_INCOME_TOTAL, na.rm = TRUE)
m_birth_train <- mean(X_train$DAYS_BIRTH, na.rm = TRUE)
m_empl_train <- mean(X_train$DAYS_EMPLOYED, na.rm = TRUE)
s_inc_train <- sd(X_train$AMT_INCOME_TOTAL)
s_birth_train <- sd(X_train$DAYS_BIRTH)
s_empl_train <- sd(X_train$DAYS_EMPLOYED)
X_train$AMT_INCOME_TOTAL <- scale(X_train$AMT_INCOME_TOTAL, center=m_inc_train, scale=s_inc_train)
X_train$DAYS_BIRTH <- scale(X_train$DAYS_BIRTH, center=m_birth_train, scale=s_birth_train)
X_train$DAYS_EMPLOYED <- scale(X_train$DAYS_EMPLOYED, center=m_empl_train, scale=s_empl_train)
X_dev$AMT_INCOME_TOTAL <- scale(X_dev$AMT_INCOME_TOTAL, center=m_inc_train, scale=s_inc_train)
X_dev$DAYS_BIRTH <- scale(X_dev$DAYS_BIRTH, center=m_birth_train, scale=s_birth_train)
X_dev$DAYS_EMPLOYED <- scale(X_dev$DAYS_EMPLOYED, center=m_empl_train, scale=s_empl_train)

### Building the model ###
#Final Setup
X_train<-data.matrix(X_train)
Y_train<- to_categorical(Y_train)
unique(NNdata$status)

#Model and Training
#Stuff we can optimize in keras_model_sequential: #layers, #units per layer, activation functions of hidden and input layers
#sigmoid for output layer and #neurons is fixed 
#Stuff we can optimize in compile: type of optimizer.
#loss function is fixed 
#Stuff we can optimize in fit: batch size, epochs

#1. Building several optimizers
#Stochastic Gradient descent
my_optimizer_sgd <- optimizer_sgd(
  learning_rate = 0.01,
  momentum = 0,
  decay = 0,
  nesterov = FALSE,
  clipnorm = NULL,
  clipvalue = NULL
)

#Momentum
my_optimizer_mom <- optimizer_sgd(
  learning_rate = 0.01,
  momentum = 0.9,
  decay = 0,
  nesterov = FALSE,
  clipnorm = NULL,
  clipvalue = NULL
)

#RMSprop
my_optimizer_rmsprop <- optimizer_rmsprop(
  learning_rate = 0.001,
  rho = 0.9,
  epsilon = NULL,
  decay = 0,
  clipnorm = NULL,
  clipvalue = NULL
)

#Adam
my_optimizer_adam <- optimizer_adam(
  learning_rate = 0.001,
  beta_1 = 0.9,
  beta_2 = 0.999,
  epsilon = NULL,
  decay = 0,
  amsgrad = FALSE,
  clipnorm = NULL,
  clipvalue = NULL
)

#Store optimizers and their name in dataframe:
#my_optimizer_vector <- c(my_optimizer_sgd, my_optimizer_mom, my_optimizer_rmsprop, my_optimizer_adam)
my_optimizer_name_vector <- c("SGD", "Momentum", "RMSprop", "ADAM")

i <- 1

while (i <= 4) {

  network <- keras_model_sequential() %>% 
    layer_dense(units = 512, activation = "relu", input_shape = c(17)) %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 9, activation = "sigmoid")
  
  if (i == 1) {
    network %>% compile(
      optimizer = my_optimizer_sgd,
      loss = "categorical_crossentropy",
      metrics = c("accuracy")
    )
  }
  if (i == 2) {
    network %>% compile(
      optimizer = my_optimizer_mom,
      loss = "categorical_crossentropy",
      metrics = c("accuracy")
    )
  }
  if (i == 3) {
    network %>% compile(
      optimizer = my_optimizer_rmsprop,
      loss = "categorical_crossentropy",
      metrics = c("accuracy")
    )
  }
  if (i == 4) {
    network %>% compile(
      optimizer = my_optimizer_rmsprop,
      loss = "categorical_crossentropy",
      metrics = c("accuracy")
    )
  }
    
  network %>% fit(X_train, Y_train, epochs = 100, batch_size = 128)
  
  print(my_optimizer_name_vector[i])
  
  i = i + 1
}
