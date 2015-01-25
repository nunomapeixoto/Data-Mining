library(DMwR)
library(rpart)
library(gbm)
library(performanceEstimation)
library(randomForest)
library(e1071)


####Submissão 1
test_data <- read.csv("data/test.csv", head=T,  na.strings=c("?"))
ar <- rpart(train_data$TotalBurntArea ~ .,train_data)
preds <- predict(ar,test_data)
res <- data.frame(names(preds), preds)
write.csv(res, row.names=F, file="Submissões/sub1.csv")


####Submissão 2
simpleBagging <- function(form,data,model='rpartXse',nModels=100,...) {
  ms <- list()
  n <- nrow(data)
  for(i in 1:nModels) {
    tr <- sample(n,n,replace=T)
    ms[[i]] <- do.call(model,c(list(form,data[tr,]),...))
  }
  ms
}
predict.simpleBagging <- function(models,test) {
  ps <- sapply(models,function(m) predict(m,test))
  apply(ps,1,mean)
}

m <- simpleBagging(TotalBurntArea ~ .,train_data,nModels=300,se=0.5)
ps <- predict.simpleBagging(m,test_data)
res2 <- data.frame(ps)
write.csv(res2, file="Submissões/sub2.csv")


####Submissão 3
m <- gbm(TotalBurntArea ~ .,,data=train_data, n.trees=5000,verbose=F)
ps <- predict(m,test_data,type='response',n.trees=5000)
res3 <- data.frame(ps)
write.csv(res3, file="Submissões/sub3.csv")



####Submissão 4
res3 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., train_data),
  workflowVariants("standardWF",
                   learner=c("rpartXse","svm","randomForest")),
  EstimationTask(metrics="mae",method=CV(nReps=2,nFolds=5)))


s1 <- svm(train_data$TotalBurntArea ~ .,train_data)
ps4 <- predict(s1,test_data)
res4 <- data.frame(ps4)
write.csv(res4, file="Submissões/sub4.csv")


####Submissão 5
res5 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., train_data),
  c(workflowVariants(learner="svm",
                     learner.pars=list(cost=c(1,10),
                                       gamma=c(0.1,0.01,1),
                                       degree=c(1,2),
                                       epsilon=0.1))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=10)))

s1 <- svm(train_data$TotalBurntArea ~ .,train_data, cost=1, gamma=0.01, epsilion=0.1, degree=2)
ps5 <- predict(s1,test_data)
res5 <- data.frame(ps5)
write.csv(res5, file="Submissões/sub5.csv")


####Submissão 6
res6 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="svm",
                     learner.pars=list(cost=c(1,10),
                                       gamma=c(0.1,0.01,1)))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=10)))


nanration <- gain.ratio(formula =  TotalBurntArea ~ ., train_data)
nanration$attribute <- rownames(nanration)
nanration <- nanration[order(nanration$attr_importance, decreasing =T ),]
nanration <- nanration[is.nan(nanration$attr_importance),]
rownames(nanration) <- 1:nrow(nanration)
att_names <- nanration$attribute
new_test_data <- test_data[,!(names(test_data) %in% att_names)]


s6 <- svm(new_train_data$TotalBurntArea ~ .,new_train_data, cost=1, gamma=0.01)
ps6 <- predict(s6,new_test_data)
res6 <- data.frame(ps6)
write.csv(res6, file="Submissões/sub6.csv")


####Submissão 7
norm.new_test_data <- cbind(scale(new_test_data[,]))
colnames(norm.new_test_data) <- colnames(new_test_data)
norm.new_test_data <- data.frame(norm.new_test_data)

res7 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., norm.new_train_data),
  c(workflowVariants(learner="svm",
                     learner.pars=list(cost=c(1,10),
                                       gamma=c(0.1,0.01,1)))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=10)))

s7 <- svm(new_train_data$TotalBurntArea ~ .,norm.new_train_data, cost=1, gamma=0.01)
ps7 <- predict(s7,norm.new_test_data)
res7 <- data.frame(ps7)
write.csv(res7, file="Submissões/sub7.csv")

####Submissão 8
norm.new_test_data <- cbind(scale(new_test_data[,]))
colnames(norm.new_test_data) <- colnames(new_test_data)
norm.new_test_data <- data.frame(norm.new_test_data)

res8 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., norm.new_train_data),
  c(workflowVariants(learner="svm",
                     learner.pars=list(cost=c(1,10),
                                       gamma=c(0.1,0.01,1)))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=10)))

s8 <- svm(TotalBurntArea ~ .,norm.new_train_data, cost=1, gamma=0.01)
ps8 <- predict(s8,norm.new_test_data)
res8 <- data.frame(ps8)
write.csv(res8, file="Submissões/sub8.csv")


####Submissão 9 
library(kernlab)
res9 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="ksvm",
                     learner.pars=list(epsilon=c(0.01,1),
                                       C=c(1,10,30),
                                       kernel=c("rbfdot","laplacedot","besseldot")))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=5)))

topPerformers(res9)

s9 <- ksvm(TotalBurntArea ~ .,new_train_data, C=1, epsilon=0.01, kernel = "rbfdot")
ps9 <- predict(s9,new_test_data)
ps9[ps9<0] <- 0
res9 <- data.frame(ps9)
write.csv(res9, file="Submissões/sub9.csv")



####Submissão 10
library(kernlab)
res9 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="ksvm",
                     learner.pars=list(epsilon=c(0.01,1),
                                       C=c(1,10,30),
                                       kernel=c("rbfdot","laplacedot","besseldot")))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=5)))

topPerformers(res9)

s10 <- ksvm(TotalBurntArea ~ .,new_train_data, C=1, epsilon=0.01, kernel = "rbfdot")
ps10 <- round(predict(s10,new_test_data))
ps10[ps10<0] <- 0
res10 <- data.frame(ps10)
write.csv(res10, file="Submissões/sub12.csv")


####Submissão 11 (Best so far)
library(kernlab)
res11 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="ksvm",
                     learner.pars=list(epsilon=c(0.01,1,10^-9),
                                       C=c(0.5, 1,2,10,30),
                                       kernel=c("rbfdot","laplacedot","besseldot")))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=5)))

topPerformers(res11)

s11 <- ksvm(TotalBurntArea ~ .,new_train_data, C=1, epsilon=10^-9, kernel = "rbfdot")
ps11 <- predict(s11,new_test_data)
ps11[ps11<0] <- 0
res11 <- data.frame(ps11)
write.csv(res11, file="Submissões/sub16.csv")


####Submissão 12
library(kknn)
res12 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="train.kknn",
                     learner.pars=list(scale=T,
                                       k=c(5,7,9, 11, 13),
                                       distance=c(1,2,3),
                                       kernel=c("epanechnikov", "triangular")))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=5)))

topPerformers(res12)


####Submissão 13
library(kernlab)
res13 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="ksvm",
                     learner.pars=list(epsilon=c(0.01,10^-7,10^-9,10^-8,10^-10),
                                       C=c(1,2,3),
                                       scaled=T,
                                       kernel=c("rbfdot")))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=10)))

topPerformers(res13)

set.seed(1234)
s13 <- ksvm(TotalBurntArea ~ .,new_train_data, C=2, epsilon=10^-10, kernel = "rbfdot", scaled=T, cross=10)
ps13 <- predict(s13,new_test_data)
ps13[ps13<0] <- 0
res13 <- data.frame(ps13)
write.csv(res13, file="Submissões/sub21.csv")



####Submissão 14

library(caret)
res14 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="nnet",
                     learner.pars=list(size=c(2,4,6),
                                       maxit=c(200,300),
                                       decay=c(0.1, 0.4),
                                       scale=T,
                                       trace = F, 
                                       linout = 1))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=5)))

topPerformers(res14)


####Submissão 15
res15 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data),
  c(workflowVariants(learner="randomForest",
                     learner.pars=list(ntree=c(250,500,1000),
                                       nodesize=c(5,10),
                                       corr.bias=c(F,T),
                                       mtry=c(3,6,9)))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=5)))

topPerformers(res15)


####Submissão 16
res13 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., new_train_data[,b]),
  c(workflowVariants(learner="ksvm",
                     learner.pars=list(epsilon=c( 10^-8, 10^-10),
                                       C=c(1,2),
                                       scaled=T,
                                       kernel=c("rbfdot")))),
  EstimationTask(metrics="mae",method=CV(nReps=6,nFolds=10)))
topPerformers(res13)

a <- gain.ratio(formula =  TotalBurntArea ~ ., new_train_data)
a$attribute <- rownames(a)
a <- head(a[order(a$attr_importance, decreasing =T ),],70)
b <- a$attribute
b<-append(b,"TotalBurntArea")

set.seed(1234)
s13 <- ksvm(TotalBurntArea ~ .,new_train_data[,b], C=2, epsilon=10^-10, kernel = "rbfdot", scaled=T, cross=10)
ps13 <- predict(s13,new_test_data)
ps13[ps13<0] <- 0
res13 <- data.frame(ps13)
write.csv(res13, file="Submissões/sub22.csv")


####Submissão 17
library(gbm)
res15 <- performanceEstimation(
  PredTask(TotalBurntArea ~ ., data=new_train_data),
  c(workflowVariants(learner="gbm.train",
                     learner.pars=list(n.trees = 100))),
  EstimationTask(metrics="mae",method=CV(nReps=3,nFolds=5)))

topPerformers(res15)

