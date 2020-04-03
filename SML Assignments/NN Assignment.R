setwd("D:\\OneDrive - IESEG\\Working\\03. MBD_SML_Statistical & ML Approaches for Mkt\\SML_Section7\\data\\com1_default")

library(neuralnet)
library(nnet)
library(mlr)

# Read data
df <- read.csv('default.csv', sep=';')
str(df)

# Encode as a one hot vector multilabel data
df2 <- cbind(df[, 2:(ncol(df)-1)], class.ind(as.factor(df$Y)))
names(df2) <- c(names(df2)[1:(ncol(df2)-2)], "N", "Y")

# Q1. ------------------------------
###### Randomly divide Train/test as 80/20 
set.seed(1)
train_idx <- sample(1:nrow(df2), round(nrow(df2)*0.8)) 
train_df2 <- df2[train_idx, ] #80
test_df2 <- df2[-train_idx, ] #20

# Create the formula
nn_formula <- as.formula(paste0('Y + N ~ ', paste(names(df2)[1:(ncol(df2)-2)], collapse=' + ')))
nn_formula

# Q2. ------------------------------
###### Build NN model with 1 hidden layer of 30 neurons
md_nnet <- neuralnet(nn_formula,
                     train_df2,
                     hidden=30,        # Number of neurons on each layer
                     #stepmax=10000,        # Maximum training step
                     rep=1,                 # Number of training repeat
                     lifesign='full',       # Print during train
                     algorithm='backprop',  # Algorithm to calculate the network
                     learningrate=0.01,     # Learning rate
                     err.fct='ce',          # Error function, cross-entropy
                     act.fct="logistic",    # Function use to calculate the result, logistic = sigmoid
                     linear.output=F
)

# Q3. ------------------------------
###### Build NN model with multiple hidden layers and sigmoid activation function
md_nnet <- neuralnet(nn_formula,
                     train_df2,
                     hidden=c(5,5),        # Number of neurons on each layer
                     #stepmax=10000,        # Maximum training step
                     rep=1,                 # Number of training repeat
                     lifesign='full',       # Print during train
                     algorithm='backprop',  # Algorithm to calculate the network
                     learningrate=0.01,     # Learning rate
                     err.fct='ce',          # Error function, cross-entropy
                     act.fct="logistic",    # Function use to calculate the result, logistic = sigmoid
                     linear.output=F
)

# Q4. ------------------------------
###### Build 5 other classification models and compare with the 2 previous NN models
###### Overall, NN is much slower than other classification models.
###### KNN was tried with different amount of neighbors (20,50,150,300,500)
###### However KNN performance was consistently around 0.77
###### For other algorithms, the performance is not that different than NN (around 0.80).

# Define the ML classification task
df$Y <- as.factor(df$Y)
train_df <- df[train_idx, ] #80
test_df <- df[-train_idx, ] #20

train_task <- mlr::makeClassifTask(id ='default_train', data=train_df, target='Y')
test_task <- mlr::makeClassifTask(id='default_test', data=test_df, target='Y')

# Logistic Regression Lasso (l1)
learner <- mlr::makeLearner('classif.LiblineaRL1LogReg')  # Register a machine learning model
model <- mlr::train(learner, train_task)
pred_test <- predict(model, task=test_task)
performance(pred_test, measures=acc)

# Decision Tree
learner <- mlr::makeLearner('classif.rpart')  # Register a machine learning model
model <- mlr::train(learner, train_task)
pred_test <- predict(model, task=test_task)
performance(pred_test, measures=acc)

# Random Forest
learner <- makeLearner('classif.randomForest')
model <- mlr::train(learner, train_task)
pred_test <- predict(model, task=test_task)
performance(pred_test, measures=acc)

# k-Nearest Neighbor (k=100)
learner <- makeLearner('classif.knn', k=100)
model <- mlr::train(learner, train_task)
pred_test <- predict(model, task=test_task)
performance(pred_test, measures=acc)

# Adabag Boosting
learner <- makeLearner('classif.boosting')
model <- mlr::train(learner, train_task)
pred_test <- predict(model, task=test_task)
performance(pred_test, measures=acc)
