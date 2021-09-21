
library(lattice)
library(ggplot2)
library(plyr)
library(randomForest)
training.raw <- read.csv("C:/Users/u557g3/Documents/Coursera/Practical Machine Learning/pml-training.csv")
testing.raw <- read.csv("C:/Users/u557g3/Documents/Coursera/Practical Machine Learning/pml-testing.csv")
# Exploratory data analyses
# Look at the dimensions & head of the dataset to get an idea

# Res 1
dim(training.raw)
## [1] 19622   160
# Res 2 - excluded because excessivness
head(training.raw)

# Res 3 - excluded because excessivness
str(training.raw)

# Res 4 - excluded because excessivness
summary(training.raw)
# What we see is a lot of data with NA / empty values. Let's remove those

maxNAPerc = 20
maxNACount <- nrow(training.raw) / 100 * maxNAPerc
removeColumns <- which(colSums(is.na(training.raw) | training.raw=="") > maxNACount)
training.cleaned01 <- training.raw[,-removeColumns]
testing.cleaned01 <- testing.raw[,-removeColumns]
# Also remove all time related data, since we won't use those

removeColumns <- grep("timestamp", names(training.cleaned01))
training.cleaned02 <- training.cleaned01[,-c(1, removeColumns )]
testing.cleaned02 <- testing.cleaned01[,-c(1, removeColumns )]
# Then convert all factors to integers

classeLevels <- levels(as.factor(training.cleaned02$classe))
training.cleaned03 <- data.frame(data.matrix(training.cleaned02))
training.cleaned03$classe <- factor(training.cleaned03$classe, labels=classeLevels)
testing.cleaned03 <- data.frame(data.matrix(testing.cleaned02))
# Finally set the dataset to be explored

training.cleaned <- training.cleaned03
testing.cleaned <- testing.cleaned03
# Exploratory data analyses
# Since the test set provided is the the ultimate validation set, we will split the current training in a test and train set to work with.

set.seed(19791108)
library(caret)

classeIndex <- which(names(training.cleaned) == "classe")

partition <- createDataPartition(y=training.cleaned$classe, p=0.75, list=FALSE)
training.subSetTrain <- training.cleaned[partition, ]
training.subSetTest <- training.cleaned[-partition, ]
# What are some fields that have high correlations with the classe?

correlations <- cor(training.subSetTrain[, -classeIndex], as.numeric(training.subSetTrain$classe))
bestCorrelations <- subset(as.data.frame(as.table(correlations)), abs(Freq)>0.25)
bestCorrelations
# Var1 Var2       Freq
# 15 magnet_belt_y    A -0.2881809
# 27  magnet_arm_x    A  0.2988049
# 28  magnet_arm_y    A -0.2633230
# 44 pitch_forearm    A  0.3370001
# Even the best correlations with classe are hardly above 0.25 Let's check visually if there is indeed hard to use these 2 as possible simple linear predictors.

library(Rmisc)
library(ggplot2)

p1 <- ggplot(training.subSetTrain, aes(classe,pitch_forearm)) + 
  geom_boxplot(aes(fill=classe))

p2 <- ggplot(training.subSetTrain, aes(classe, magnet_arm_x)) + 
  geom_boxplot(aes(fill=classe))

multiplot(p1,p2,cols=2)


# Clearly there is no hard seperation of classes possible using only these 'highly' correlated features. Let's train some models to get closer to a way of predicting these classe's
# 
# Model selection
# Let's identify variables with high correlations amongst each other in our set, so we can possibly exclude them from the pca or training.
# 
# We will check afterwards if these modifications to the dataset make the model more accurate (and perhaps even faster)

library(corrplot)
correlationMatrix <- cor(training.subSetTrain[, -classeIndex])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9, exact=TRUE)
excludeColumns <- c(highlyCorrelated, classeIndex)
corrplot(correlationMatrix, method="color", type="lower", order="hclust", tl.cex=0.70, tl.col="black", tl.srt = 45, diag = FALSE)


# We see that there are some features that aree quite correlated with each other. We will have a model with these excluded. Also we'll try and reduce the features by running PCA on all and the excluded subset of the features

pcaPreProcess.all <- preProcess(training.subSetTrain[, -classeIndex], method = "pca", thresh = 0.99)
training.subSetTrain.pca.all <- predict(pcaPreProcess.all, training.subSetTrain[, -classeIndex])
training.subSetTest.pca.all <- predict(pcaPreProcess.all, training.subSetTest[, -classeIndex])
testing.pca.all <- predict(pcaPreProcess.all, testing.cleaned[, -classeIndex])


pcaPreProcess.subset <- preProcess(training.subSetTrain[, -excludeColumns], method = "pca", thresh = 0.99)
training.subSetTrain.pca.subset <- predict(pcaPreProcess.subset, training.subSetTrain[, -excludeColumns])
training.subSetTest.pca.subset <- predict(pcaPreProcess.subset, training.subSetTest[, -excludeColumns])
testing.pca.subset <- predict(pcaPreProcess.subset, testing.cleaned[, -classeIndex])
# Now we'll do some actual Random Forest training. We'll use 200 trees, because I've already seen that the error rate doesn't decline a lot after say 50 trees, but we still want to be thorough. Also we will time each of the 4 random forest models to see if when all else is equal one pops out as the faster one.

library(randomForest)

ntree <- 200 #This is enough for great accuracy (trust me, I'm an engineer). 

start <- proc.time()
rfMod.cleaned <- randomForest(
  x=training.subSetTrain[, -classeIndex], 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest[, -classeIndex], 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) #do.trace=TRUE
proc.time() - start
# user  system elapsed 
# 67.86    0.78   68.71  
start <- proc.time()
rfMod.exclude <- randomForest(
  x=training.subSetTrain[, -excludeColumns], 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest[, -excludeColumns], 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) #do.trace=TRUE
proc.time() - start
# user  system elapsed 
# 67.86    0.78   68.71 
start <- proc.time()
rfMod.pca.all <- randomForest(
  x=training.subSetTrain.pca.all, 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest.pca.all, 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) #do.trace=TRUE
proc.time() - start
# user  system elapsed 
# 70.89    1.16   73.00 
start <- proc.time()
rfMod.pca.subset <- randomForest(
  x=training.subSetTrain.pca.subset, 
  y=training.subSetTrain$classe,
  xtest=training.subSetTest.pca.subset, 
  ytest=training.subSetTest$classe, 
  ntree=ntree,
  keep.forest=TRUE,
  proximity=TRUE) #do.trace=TRUE
proc.time() - start
# user  system elapsed 
# 65.64    1.08   66.74 
# Model examination
# Now that we have 4 trained models, we will check the accuracies of each. (There probably is a better way, but this still works good)

rfMod.cleaned
# Call:
#   randomForest(x = training.subSetTrain[, -classeIndex], y = training.subSetTrain$classe,      xtest = training.subSetTest[, -classeIndex], ytest = training.subSetTest$classe,      ntree = ntree, proximity = TRUE, keep.forest = TRUE) 
# Type of random forest: classification
# Number of trees: 200
# No. of variables tried at each split: 7
# 
# OOB estimate of  error rate: 0.28%
# Confusion matrix:
#   A    B    C    D    E  class.error
# A 4184    0    0    0    1 0.0002389486
# B    5 2841    2    0    0 0.0024578652
# C    0   10 2557    0    0 0.0038955980
# D    0    0   17 2395    0 0.0070480929
# E    0    0    0    6 2700 0.0022172949
# Test set error rate: 0.31%
# Confusion matrix:
#   A   B   C   D   E class.error
# A 1395   0   0   0   0 0.000000000
# B    4 942   3   0   0 0.007376185
# C    0   2 853   0   0 0.002339181
# D    0   0   1 802   1 0.002487562
# E    0   0   0   4 897 0.004439512
rfMod.cleaned.training.acc <- round(1-sum(rfMod.cleaned$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.cleaned.training.acc)
## [1] "Accuracy on training: 0.984"
rfMod.cleaned.testing.acc <- round(1-sum(rfMod.cleaned$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.cleaned.testing.acc)
## [1] "Accuracy on testing: 0.983"
rfMod.exclude
# Call:
#   randomForest(x = training.subSetTrain[, -excludeColumns], y = training.subSetTrain$classe,      xtest = training.subSetTest[, -excludeColumns], ytest = training.subSetTest$classe,      ntree = ntree, proximity = TRUE, keep.forest = TRUE) 
# Type of random forest: classification
# Number of trees: 200
# No. of variables tried at each split: 6
# 
# OOB estimate of  error rate: 0.28%
# Confusion matrix:
#   A    B    C    D    E  class.error
# A 4184    1    0    0    0 0.0002389486
# B    4 2842    2    0    0 0.0021067416
# C    0   12 2555    0    0 0.0046747176
# D    0    0   15 2396    1 0.0066334992
# E    0    0    0    6 2700 0.0022172949
# Test set error rate: 0.29%
# Confusion matrix:
#   A   B   C   D   E class.error
# A 1395   0   0   0   0 0.000000000
# B    2 945   2   0   0 0.004214963
# C    0   4 851   0   0 0.004678363
# D    0   0   3 800   1 0.004975124
# E    0   0   0   2 899 0.002219756
rfMod.exclude.training.acc <- round(1-sum(rfMod.exclude$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.exclude.training.acc)
## [1] "Accuracy on training: 0.984"
rfMod.exclude.testing.acc <- round(1-sum(rfMod.exclude$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.exclude.testing.acc)
## [1] "Accuracy on testing: 0.984"
rfMod.pca.all
# Call:
#   randomForest(x = training.subSetTrain.pca.all, y = training.subSetTrain$classe,      xtest = training.subSetTest.pca.all, ytest = training.subSetTest$classe,      ntree = ntree, proximity = TRUE, keep.forest = TRUE) 
# Type of random forest: classification
# Number of trees: 200
# No. of variables tried at each split: 6
# 
# OOB estimate of  error rate: 2.08%
# Confusion matrix:
#   A    B    C    D    E class.error
# A 4162   11    1    8    3 0.005495818
# B   49 2775   18    2    4 0.025632022
# C    5   30 2509   21    2 0.022594468
# D    3    1  100 2300    8 0.046434494
# E    1    7   14   18 2666 0.014781966
# Test set error rate: 1.71%
# Confusion matrix:
#   A   B   C   D   E class.error
# A 1387   4   0   3   1 0.005734767
# B   14 930   5   0   0 0.020021075
# C    1  11 836   6   1 0.022222222
# D    1   0  23 777   3 0.033582090
# E    0   0   7   4 890 0.012208657
rfMod.pca.all.training.acc <- round(1-sum(rfMod.pca.all$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.pca.all.training.acc)
## [1] "Accuracy on training: 0.885"
rfMod.pca.all.testing.acc <- round(1-sum(rfMod.pca.all$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.pca.all.testing.acc)
## [1] "Accuracy on testing: 0.906"
rfMod.pca.subset
# Call:
#   randomForest(x = training.subSetTrain.pca.subset, y = training.subSetTrain$classe,      xtest = training.subSetTest.pca.subset, ytest = training.subSetTest$classe,      ntree = ntree, proximity = TRUE, keep.forest = TRUE) 
# Type of random forest: classification
# Number of trees: 200
# No. of variables tried at each split: 6
# 
# OOB estimate of  error rate: 2.43%
# Confusion matrix:
#   A    B    C    D    E class.error
# A 4155    8    9   10    3 0.007168459
# B   69 2741   29    5    4 0.037570225
# C    2   33 2510   20    2 0.022204908
# D    7    3  104 2292    6 0.049751244
# E    2    9   20   13 2662 0.016260163
# Test set error rate: 1.96%
# Confusion matrix:
#   A   B   C   D   E class.error
# A 1387   3   2   3   0 0.005734767
# B   18 924   7   0   0 0.026343519
# C    0  12 838   4   1 0.019883041
# D    0   1  27 774   2 0.037313433
# E    1   3   9   3 885 0.017758047
rfMod.pca.subset.training.acc <- round(1-sum(rfMod.pca.subset$confusion[, 'class.error']),3)
paste0("Accuracy on training: ",rfMod.pca.subset.training.acc)
## [1] "Accuracy on training: 0.867"
rfMod.pca.subset.testing.acc <- round(1-sum(rfMod.pca.subset$test$confusion[, 'class.error']),3)
paste0("Accuracy on testing: ",rfMod.pca.subset.testing.acc)
## [1] "Accuracy on testing: 0.893"
# Conclusion
# This concludes that nor PCA doesn't have a positive of the accuracy (or the process time for that matter) The rfMod.exclude perform's slightly better then the 'rfMod.cleaned'
# 
# We'll stick with the rfMod.exclude model as the best model to use for predicting the test set. Because with an accuracy of 98.7% and an estimated OOB error rate of 0.23% this is the best model.

# Before doing the final prediction we will examine the chosen model more in depth using some plots

par(mfrow=c(1,2)) 
varImpPlot(rfMod.exclude, cex=0.7, pch=16, main='Variable Importance Plot: rfMod.exclude')
plot(rfMod.exclude, , cex=0.7, main='Error vs No. of trees plot')
par(mfrow=c(1,1)) 
##     user   system  elapsed 
## 4832.341   57.977 4936.684
# Test results
# Although we've chosen the rfMod.exclude it's still nice to see what the other 3 models would predict on the final test set. Let's look at predictions for all models on the final test set.

predictions <- t(cbind(
  exclude=as.data.frame(predict(rfMod.exclude, testing.cleaned[, -excludeColumns]), optional=TRUE),
  cleaned=as.data.frame(predict(rfMod.cleaned, testing.cleaned), optional=TRUE),
  pcaAll=as.data.frame(predict(rfMod.pca.all, testing.pca.all), optional=TRUE),
  pcaExclude=as.data.frame(predict(rfMod.pca.subset, testing.pca.subset), optional=TRUE)
))
predictions
##            1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20 
## exclude    "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"
## cleaned    "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"
## pcaAll     "B" "A" "C" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"
## pcaExclude "B" "A" "C" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"
# The predictions don't really change a lot with each model, but since we have most faith in the rfMod.exclude,we'll keep that as final answer.