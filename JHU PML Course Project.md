Human Activity Recognition: Predicting quality of dumbbell lift execution
========================================================
Course Project submission for [JHU Practical Machine Learning](https://class.coursera.org/predmachlearn-005)

Irene Rix, 20 September 2014

## Introduction

This document describes a predictive model built to determine the quality of Unilateral Dumbbell Bicep Curls using sensor data. The dataset is accessed under the Creative Commons License (CC BY-SA) from the following source:

<cite>Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.</cite>

The data comprise a labelled target variable, 'classe', and a range of measurements from sensors on subjects' arms, forearms, and belt as well as on the dumbbell itself. The target variable 'classe' has 5 categories: Class A being perfect execution, and classes B, C, D and E being common errors.

The objective of this model is to accurately predict the quality of execution, 'classe', using the various sensor readings.

## Set up

Import the full raw dataset, before splitting the data into training and test sets using a 70/30 split. The test set, 'test', will be set aside for the entirety of the model development process, while the training dataset, 'train', will be subject to various train/validation splits throughout the model building process.

```r
# get full data
fulldata <- read.table('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', header=T, sep=",")  # read from url
# fulldata <- read.table('pml-training.csv', header=T, sep=",")  # read from disk

# set seed for reproducability
set.seed(42)

# partition the training and test data
require(caret, quiet=T)
```

```
## Warning: package 'caret' was built under R version 3.0.3
## Warning: package 'lattice' was built under R version 3.0.3
## Warning: package 'ggplot2' was built under R version 3.0.3
```

```r
partition <- createDataPartition(y=fulldata$classe ,p=0.7, list=F)
train <- fulldata[partition,]; test <- fulldata[-partition,]

# review dimensions of training and hold-out test data
dim(train); dim(test)
```

```
## [1] 13737   160
```

```
## [1] 5885  160
```

## Data hygiene

A large fraction of the variables in the raw data show little variation or are dominated by NA/missing values - both of which can negatively impact modelling efforts. Identify these fields and remove them from further analysis.

```r
# identify variables with near zero variance
nzv <- nearZeroVar(train, saveMetrics=F)
nzv <- names(train[nzv])

# identify variables with a high proportion of missing data
nas <- apply(train, 2, function(x) sum(is.na(x))) > (0.75 * nrow(train))
nas <- names(nas)[nas==T]

# create list of valid variables, cut down training data, and check new dimensions
keepers <- setdiff(names(train), c(nzv, nas))
train <- train[keepers]; dim(train)
```

```
## [1] 13737    59
```

The data includes a number of session variables such as  'user_name', 'num_window', and several timestamps. While some of these variables are highly predictive on the current data, they would be unlikely to generalise to future data, for example when collecting sensor data from some new, unknown person on a date not included in the current range.

Similarly the case index reads into R as 'X' and is almost perfectly predictive of classe, but cannot be expected to be available for future data.

As such, these session variables are excluded and prediction is based on the sensor data alone. This is at the cost of a small margin of accuracy on the training and current test data, but will yield a more reliable model overall.


```r
# remove session variables that will not generalise to future observations
keepers <- setdiff(names(train), c("X", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", 'num_window', 'user_name'))
train <- train[keepers]; dim(train)
```

```
## [1] 13737    53
```

Histograms, and boxplots cut by the target variable, were generated for the remaining numeric variables. The commented code below writes many charts to disk for review; these were manually inspected to better understand the distributions of the variables and their relationship with the target. 

```r
# # subset to numeric independent variables
# X <- train[unlist(lapply(train, class)) %in% c('integer', 'numeric')]
#
# # save histograms
# for (i in 1:ncol(X)){
#     jpeg(file=paste("plots/hist", names(X)[i], ".jpg", sep=""))
#     hist(X[[i]], main=names(X)[i])
#     dev.off()
# }
# 
# # save box plots against classe
# for (i in 1:ncol(X)){
#   jpeg(file=paste("plots/box", names(X)[i], ".jpg", sep=""))
#   plot(training$classe, X[[i]], main=names(X)[i])
#   dev.off()
# }
```

Inspection of the plots revealed many non-normal distributions, with some bi-modal and others heavily skewed, as well as the presence of outliers and missing values in the data. 

While many of the statistical outliers were information-bearing and thus retained, one record showed exceptionally extreme values that suggest a data error of some sort. A single example is shown (and circled in red) below, but a similar pattern was evident across all 6 of the gyros_dumbbell and gyros_forearm measures.

```r
# plot
plot(train$classe, train[['gyros_dumbbell_z']], main='gyros_dumbbell_z') 

# circle outlier
points(train$class[which.max(train$gyros_dumbbell_z)], train[['gyros_dumbbell_z']][which.max(train$gyros_dumbbell_z)], col='red', cex=5) 
```

![plot of chunk boxPlot](figure/boxPlot.png) 

On confirmation that all 6 of these extreme results are associated with the same row of data, the record is assumed to be bad data and thrown out for the remainder of analysis:

```r
# identify extreme outlier for one of the variables
rogue <- which.max(train$gyros_dumbbell_z)

# check this record's values and impact on population min/max
gyr <- train[c('gyros_dumbbell_x', 'gyros_dumbbell_y', 'gyros_dumbbell_z', 'gyros_forearm_x', 'gyros_forearm_y', 'gyros_forearm_z')]
x <- rbind(gyr[rogue,], apply(gyr, 2, min), apply(gyr, 2, max), apply(gyr[-rogue,], 2, min), apply(gyr[-rogue,], 2, max))
rownames(x) <- c(paste0('case #', rogue), 'min all', 'max all', paste0('min without case #', rogue), paste0('max without case #', rogue))
x
```

```
##                        gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z
## case #3775                      -204.00            52.00           317.00
## min all                         -204.00            -2.10            -2.30
## max all                            2.22            52.00           317.00
## min without case #3775            -1.94            -2.10            -2.30
## max without case #3775             2.22             2.73             1.87
##                        gyros_forearm_x gyros_forearm_y gyros_forearm_z
## case #3775                      -22.00          311.00          231.00
## min all                         -22.00           -6.52           -8.09
## max all                           3.97          311.00          231.00
## min without case #3775           -3.36           -6.52           -8.09
## max without case #3775            3.97            6.13            4.10
```

```r
# remove extreme outlier case
train <- train[-rogue]
```

# Model build

### Approach
#### Treatment of timed observations
Although the sensor data could be considered a time series, the 20 cases provided for assessment are not contiguous. Therefore the time series approach was rejected and each record is treated as independent in the following analyses.

##### Algorithm
With a large number of potentially correlated, non-normally-distributed predictors and a multi-category target variable, Random Forests was selected as the initial prediction algorithm. Performance was so strong that no further algorithms were incorporated.

##### Cross validation and tuning
The training data (itself 70% of the original dataset), was split into 10 folds using the K-folds approach. To optimise the number of variables per tree, various levels were tested across multiple model builds and the resulting accuracy plotted. Each build in turn comprised 10 cycles of model/cross-validation (one for each of the 10 folds). 

#### Transformations
Mean normalisation was not necessary due to the selection of Random Forests as the modelling approach: Random Forests is largely unaffected by monotonic changes to the independent variables and as such feature scaling and centering are not required. Principle Component Analysis (PCA) was considered for data reduction purposes but rejected due to its potential sensitivity to the remaining outliers. 

### Preparation
Split the testing dataset into k-folds for use in model tuning. Import the library for random forests.

```r
# create folds
nfolds <- 10
folds <- createFolds(y=train$classe, k=nfolds, list=F, returnTrain=T)
require(randomForest, quiet=T)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

Create function to build and cross-validate across each of the K folds. The function accepts parameters for the number of trees and number of variables in each tree, runs and cross-validates for each fold, and returns the average accuracy across the K models.

```r
# function to loop through each fold running model with the given parameters
modelfolds <- function(ntrees, nvars){
  results <- c()
  for (i in 1:nfolds){
    # set data
    temptrain <- train[folds!=i,]
    tempval <- train[folds==i,]
    # train model
    tempmodel <- randomForest(classe ~ ., data=temptrain, ntree=ntrees, mtry=nvars)
    # validate
    tempresult <- predict(tempmodel, tempval)==tempval$classe
    results <- c(results, tempresult)
  }
  mean(results)  # return the mean accuracy across the the 10 folded models
}
```

### Optimise number of variables per tree
Test the model with different numbers of variables to identify the optimal number for use in the final model. Complete 5 builds for each value of mtry in the "vartests" obect and record accuracy in the "varmatrix" object.

```r
# create vector of possible values for mtry, 'vartests', and set number of builds for each test, 'r'
vartests <- c(3, seq(5, 30, 5)); r <- 5;

# create matrix to house results
varmatrix <- matrix(nrow=length(vartests), ncol=r); rownames(varmatrix) <- vartests; colnames(varmatrix) <- 1:r

# test each level of vartests across the K folds using the 'modelfolds' function. Repeat 'r' times.
for (i in 1:length(vartests)){
  for (ii in 1:r){
    varmatrix[i, ii] <- modelfolds(ntrees=100, nvars=vartests[i])  # save results of model with this parameter
  }
}
```

Plot model accuracy achieved with different numbers of variables in the trees. Points represent accuracy within a set of K models; the blue line represents the average across all repeats.

Result: The curved result suggests some overfitting may occur with 20 or more variables in trees. Opted for 10 variables per tree for the final model.

```r
# plot points from the results matrix
plot(rep(rownames(varmatrix), r),as.vector(varmatrix), xlab='Number of predictors', ylab='Average accuracy', main='Accuracy by Number of Predictors')

# take mean for each level and plot as a line
varresult <- apply(varmatrix, 1, mean) # take average across runs
points(as.integer(names(varresult)), varresult, type='l', col='blue')
```

![plot of chunk plotNvars](figure/plotNvars.png) 

### Full model build
Build a model on the full training data using the mtry parameter identified above mtry=10 and ntree=1000 (larger ntree used to reduce variance in final mode - verify that this is sufficient by plotting the model and examining the error rates by number of trees). 

```r
# build model
modelfinal <- randomForest(classe ~ ., data=train, ntree=500, mtry=10, proximity=T)

# examine model
modelfinal
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train, ntree = 500,      mtry = 10, proximity = T) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 10
## 
##         OOB estimate of  error rate: 0.5%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    3    0    0    1    0.001024
## B   12 2645    1    0    0    0.004891
## C    0   12 2382    2    0    0.005843
## D    0    0   27 2224    1    0.012433
## E    0    1    2    6 2516    0.003564
```

```r
# plot model to check ntree parameter was adequate
plot(modelfinal)
```

![plot of chunk finalModel](figure/finalModel.png) 

Result: the error rates drops rapidly and have stabilised well before the 1000 tree point. This suggests that 1000 trees is more than sufficient for this model and no further adjustments are necessary.

## Test final model

#### Out of sample error
Test on the 30% testing sample 'test' which was set aside at the start, for an indication of out-of-sample error.

```r
# predict on 30% hold-out sample
testpred <- predict(modelfinal, test[keepers])

# verify results for additional OOS estimate
table(testpred, test$classe)
```

```
##         
## testpred    A    B    C    D    E
##        A 1673    7    0    0    0
##        B    1 1131    7    0    0
##        C    0    1 1016   10    0
##        D    0    0    3  954    0
##        E    0    0    0    0 1082
```

```r
mean(testpred==test$classe)
```

```
## [1] 0.9951
```


#### Submission cases
Run on the 20 grading cases and obtain predictions for submission

```r
# predict for the 20 submission cases
grading <- read.table('http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', header=T, sep=",")[setdiff(keepers, 'classe')]
gradepred <- predict(modelfinal, grading)

# # write out text files to 'submissions' folder for submission
# for (i in 1:length(gradepred)){
#   write.table(gradepred[i], paste0("submissions\\id ", i, ".txt"), row.names=F, col.names=F, quote=F)
# }

# print results
gradepred
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

## Interpretation
#### Goodness of fit
The overall model fit is very strong, with low errors on both the training and testing sets.

The OOB estimate of error was 0.50 and the model correctly classified 99.51% of the cases in the "test" set, suggesting that the out-of-sample accuracy will be high. However despite excluding session variables such as subject name and time, the test data still came from the same original data source so performance on data collected separately will likely be significantly lower than that observed here.

The margin by class plot below shows that prediction had the strongest margins among the Class A category, which represented a correct execution of the dumbbell lift, while margins were slightly lower for some of the poor execution groups (e.g. Class B). 

The high performance on Class A is promising as it suggests that the model will be robust in its identification of accurate vs. non accurate execution of the exercise (i.e. overall quality of execution).

```r
plot(as.factor(names(margin(modelfinal))), margin(modelfinal), xlab='Actual classe', ylab='Margin', main='Margin by Classe')
```

![plot of chunk plotMargin](figure/plotMargin.png) 
#### Variable importance

The variable importance chart shows the relative importance of the predictors in determining the way in which the exercise was performed. In this case, the most important predictors were the variables named roll_belt, pitch_forearm, yaw_belt, magnet_dumbbell_z, magnet_dumbbell_y, pitch_belt and roll_forearm.

This suggests that the sensors at the belt, forearm and on the dumbbell are the most important in determining execution quality, while the sensor on the arm is less predictive. This hypothesis could be tested by fitting the model without the 'arm' sensor data and comparing performance. 


```r
varImpPlot(modelfinal)
```

![plot of chunk plotImportance](figure/plotImportance.png) 


