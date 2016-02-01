Practical Machine Learning - Course Project
========================================================

S. A. Batla 2016 January 31

**Background**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

**Data**

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

**Goal**

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

# Load necessary libraries


```r
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)
```



# Load Training and Test Data

```r
# first, set.seed for consistency and reproducibility
set.seed(0509)

trainingSet <- read.csv("data/pml-training.csv",header=TRUE, na.strings=c("NA","#DIV/0!",""))
submissionTestSet <- read.csv("data/pml-testing.csv",header=TRUE, na.strings=c("NA","#DIV/0!",""))

dim(trainingSet)
```

```
## [1] 19622   160
```

```r
dim(submissionTestSet)
```

```
## [1]  20 160
```

# Cleaning the training data


```r
nzv <- nearZeroVar(trainingSet, saveMetrics=TRUE)
trainingSet <- trainingSet[,nzv$nzv==FALSE]

nzv <- nearZeroVar(submissionTestSet, saveMetrics=TRUE)
submissionTestSet <- submissionTestSet[,nzv$nzv==FALSE]


dim(trainingSet)
```

```
## [1] 19622   124
```

```r
dim(submissionTestSet)
```

```
## [1] 20 59
```

```r
# after trying several thresholds, it seemed a threshold of above .95 is where
# any significant change in # of dims removed would occur. May revisit if
# predictions suffer
uselessDims <- sapply(trainingSet, function(x) mean(is.na(x))) > 0.95
trainingSet <- trainingSet[, uselessDims==F]

uselessDims <- sapply(submissionTestSet, function(x) mean(is.na(x))) > 0.95
submissionTestSet <- submissionTestSet[, uselessDims==F]

# remove first few variables as they will not help with the model
trainingSet <- trainingSet[, -(1:5)]
submissionTestSet <- submissionTestSet[, -(1:5)]
```

Now find and keep highly correlated variables, try with cutoff=.75
The idea here is to reduce the number of irrelevant variables for training the model


```r
corMatrix <- cor(trainingSet[,1:53])

highlyCorrelated <- findCorrelation(corMatrix, cutoff=0.75)
trainingSet <- trainingSet[,-(highlyCorrelated)]
submissionTestSet <- submissionTestSet[,-(highlyCorrelated)]

dim(trainingSet)
```

```
## [1] 19622    33
```

```r
dim(submissionTestSet)
```

```
## [1] 20 33
```

# Partitioning the training data

Since the training set is large (19,622 records) and the test set for submission 
is very small (20 records), I will partition the training set with a 60/40 
training to test set


```r
trainingSplit <- createDataPartition(trainingSet$classe, p=0.6, list=FALSE)
trainingSubset <- trainingSet[trainingSplit, ]
testingSubset <- trainingSet[-trainingSplit, ]
dim(trainingSubset); dim(testingSubset)
```

```
## [1] 11776    33
```

```
## [1] 7846   33
```

# Developing the Model

Random Forest model was chosen first since there is "no need for cross-validation
or a separate test set to get an unbiased estimate of the test of the error".
From: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm


```r
set.seed(0509)
# 10 fold cross-validation 
fitControl <- trainControl(method="cv", number=10, allowParallel=TRUE, verboseIter = FALSE)

# Now fit the model on the training subset
fit <- train(classe ~ ., data=trainingSubset, method="rf", trcontrol=fitControl, proxy=FALSE)
```

```r
print(fit)
```

```
## Random Forest 
## 
## 11776 samples
##    32 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9903643  0.9878043  0.001672329  0.002114727
##   17    0.9952904  0.9940395  0.001470023  0.001861296
##   32    0.9899829  0.9873245  0.002423290  0.003054144
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 17.
```

```r
print(fit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, trcontrol = ..1,      proxy = FALSE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 17
## 
##         OOB estimate of  error rate: 0.25%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3344    3    0    0    1 0.001194743
## B    3 2271    5    0    0 0.003510312
## C    0    6 2044    4    0 0.004868549
## D    0    1    3 1926    0 0.002072539
## E    0    0    0    3 2162 0.001385681
```

The OOB estimate of error rate is very low, therefore I decided to next apply the model to the testing subset partitioned from original training set.


```r
predictTrainingSubset <- predict(fit, newdata=testingSubset)
confusionMatrix(testingSubset$classe, predictTrainingSubset)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    3 1507    7    0    1
##          C    0    4 1363    1    0
##          D    0    0    7 1279    0
##          E    0    1    0    3 1438
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9966         
##                  95% CI : (0.995, 0.9977)
##     No Information Rate : 0.2849         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9956         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9967   0.9898   0.9969   0.9993
## Specificity            1.0000   0.9983   0.9992   0.9989   0.9994
## Pos Pred Value         1.0000   0.9928   0.9963   0.9946   0.9972
## Neg Pred Value         0.9995   0.9992   0.9978   0.9994   0.9998
## Prevalence             0.2849   0.1927   0.1755   0.1635   0.1834
## Detection Rate         0.2845   0.1921   0.1737   0.1630   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9993   0.9975   0.9945   0.9979   0.9993
```
The confusion matrix shows an accuracy of > 99%. I will not bother with other approaches such as Decision Tree or Generalized Boosted Regression. This is very good accuracy.

I will now use this model to predict the exercise behavior (classe) of the submission test data set.


```r
# Now test on 20 test rows using RF model
predictSubmissionTestSet <- predict(fit, newdata=submissionTestSet)
print(predictSubmissionTestSet)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

