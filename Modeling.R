
origin <- "C:/Users/mkirkpatrick/National University/Precision Institute - Documents/Data Science/Modeling/Student Starts/Saved Objects/"



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                   HAZARD MODEL                                   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
dftrainHAZu <- read.csv(paste0(origin,"dftrainHAZunbalanced.csv"), na.strings="NA", stringsAsFactors=TRUE)
dftrainHAZb <- read.csv(paste0(origin,"dftrainHAZbalanced.csv"), na.strings="NA", stringsAsFactors=TRUE)


library(survival)
library(survminer)

#~~~~~~~~~~~~~~~~~~~~~~~~#
#     MODEL TRAINING     #
#~~~~~~~~~~~~~~~~~~~~~~~~#
haz.u <- coxph(Surv(daysToEvent, eventStartYN) ~ . , data = dftrainHAZu)
save(haz.u,file = paste0(origin,"hazu.rda"))

haz.b <- coxph(Surv(daysToEvent, eventStartYN) ~ . , data = dftrainHAZb)
save(haz.b,file = paste0(origin,"hazb.rda"))


#~~~~~~~~~~~~~~~~~~~~~~~#
#     MODEL TESTING     #
#~~~~~~~~~~~~~~~~~~~~~~~#
library(survival)
library(caret)
library(pROC)
library(pec)

f.hazModelEval <- function(mdel, testdf, time){
  df <- testdf
  df$P <- as.numeric(predictSurvProb(mdel, newdata = df, times = time))
  df$PCat <- 0
  df$PCat[df$P <= 0.5 ] <- 1 # less than 0.5 because P is the predicted prob of survival, so survival represents not starting
  auc.haz <- roc(df$eventStartYN, df$P)
  cm.haz <- confusionMatrix(df$eventStartYN, df$PCat, positive = "1")
  out <- list("AUC" = auc.haz, "cm" = cm.haz)
  return(out)
}


load(paste0(origin,"hazb.rda"))
load(paste0(origin,"hazu.rda"))
dftestHAZ <- read.csv(paste0(origin,"dftestHAZ.csv"), na.strings="NA", stringsAsFactors=TRUE)
dftrainHAZu <- read.csv(paste0(origin,"dftrainHAZunbalanced.csv"), na.strings="NA", stringsAsFactors=TRUE)
dftrainHAZb <- read.csv(paste0(origin,"dftrainHAZbalanced.csv"), na.strings="NA", stringsAsFactors=TRUE)

time = 380
res.haz.b <- f.hazModelEval(haz.b, dftestHAZ, time)
res.haz.u <- f.hazModelEval(haz.u, dftestHAZ, time)

rm(haz.b, haz.u, dftestHAZ, dftrainHAZu, dftrainHAZb)

save(res.haz.b,file = paste0(origin,"res.haz.b.rda"))
save(res.haz.u,file = paste0(origin,"res.haz.u.rda"))






# USING STANDARD PREDICT FUNCTION GIVES SAME RESULTS AS 380 DAYS (i.e., full data set)
# library(survival)
# library(caret)
# library(pROC)
# 
# ## See predict.coxph {survival} for documentation
# # risk < 1 = NO for eventStartYN
# # risk > 1 = YES for eventStartYN
# dftestHAZ$risk <- predict(haz, newdata = dftestHAZ, type = "risk")
# dftestHAZ$riskCat <- 0
# dftestHAZ$riskCat[dftestHAZ$risk > 1 ] <- 1
# 
# auc.haz <- roc(dftestHAZ$eventStartYN, dftestHAZ$risk)
# cm.haz <- confusionMatrix(dftestHAZ$eventStartYN, dftestHAZ$riskCat)
# auc.haz$auc
# cm.haz
# plot(auc.haz)
# 
# 
# # TRY PREDICTIONS FOR 100 DAYS
# test <- subset(dftestHAZ, daysToEvent <=100)
# test$P <- as.numeric(predictSurvProb(haz, newdata = test, times = 100))
# test$PCat <- 0
# test$PCat[test$P <= 0.5 ] <- 1 # less than 0.5 because P is the predicted prob of survival, so survival represents not starting
# 
# auc.haz <- roc(test$eventStartYN, test$P)
# cm.haz <- confusionMatrix(test$eventStartYN, test$PCat)
# auc.haz$auc
# cm.haz
# plot(auc.haz)



# ############ THIS TAKES FOREVER TO RUN.  Keeping just as a reference.
# # Incident case/dynamic control ROC/AUC using risksetROC package (Heagerty, 2005)
# # Cases are those who died at time t (incident cases).
# # Controls are those who survived until time t (dynamic controls).
# 
# # We want C/D (cumulative case/dynamic control) because we want all events (starts) prior to time t to be considered starts
# # We don't want I/D (incident sensitivity and dynamic specificity) because that 
# 
# # Example Code
# ## Define a function
# fun.survivalROC <- function(lp, t) {
#   res <- with(pbc,
#               survivalROC(Stime        = time,
#                           status       = event,
#                           marker       = get(lp),
#                           predict.time = t,
#                           method       = "KM"))       # KM method without smoothing
#   
#   ## Plot ROCs
#   with(res, plot(TP ~ FP, type = "l", main = sprintf("t = %.0f, AUC = %.2f", t, AUC)))
#   abline(a = 0, b = 1, lty = 2)
#   
#   res
# }
# 
# ## 2 x 5 layout
# layout(matrix(1:10, byrow = T, ncol = 5))
# 
# ## Model with age and sex
# res.survivalROC.age.sex <- lapply(1:10 * 365.25, function(t) {
#   fun.survivalROC(lp = "lp.age.sex", t)
# })
# 
# ## Model with age, sex, and albumin
# res.survivalROC.age.sex.albumin <- lapply(1:10 * 365.25, function(t) {
#   fun.survivalROC(lp = "lp.age.sex.albumin", t)
# })
# 
# # MY CODE
# dftest <- dftestHAZ[1:100,]
# 
# # THE linear prediction marker might be able to be converted to a simpl Y/N
# res <- with(dftest,
#             survivalROC(Stime        = daysToEvent,
#                         status       = eventStartYN,
#                         marker       = lp,
#                         predict.time = 365,
#                         method       = "KM")) 
# 
# with(res, plot(TP ~ FP, type = "l"))
# abline(a = 0, b = 1, lty = 2)
# 
# res




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                   MACHINE LEARNING                                   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#                        
dftrainMLb <- read.csv(paste0(origin,"dftrainMLbalanced.csv"), na.strings="NA", stringsAsFactors=TRUE)
dftrainMLu <- read.csv(paste0(origin,"dftrainMLunbalanced.csv"), na.strings="NA", stringsAsFactors=TRUE)

library(caret)
library(parallel)
library(doParallel)

# INITIATE PARALLEL PROCESSING
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# DEFINE TRAIN CONTROL
# use twoClassSummary so we evaluate the model wit ROC instead of Accuracy
tc <- trainControl(method="cv", number=10, savePredictions = "all", classProbs=T, summaryFunction=twoClassSummary, allowParallel = T)


#~~~~~~~~~~~~~~~~~~~~~~~~#
#     MODEL TRAINING     #
#~~~~~~~~~~~~~~~~~~~~~~~~#
lr.b <- train(eventStartYN ~ . , data=dftrainMLb, method="glm", family="binomial", trControl=tc, preProc = c("center","scale"))
save(lr.b,file = paste0(origin,"lrb.rda"))
lr.u <- train(eventStartYN ~ . , data=dftrainMLu, method="glm", family="binomial", trControl=tc, preProc = c("center","scale"))
save(lr.u,file = paste0(origin,"lru.rda"))

xgb.b <- train(eventStartYN ~ ., data=dftrainMLb, method="xgbTree", trControl=tc, preProc = c("center","scale"))
save(xgb.b,file = paste0(origin,"xgbb.rda"))
xgb.u <- train(eventStartYN ~ ., data=dftrainMLu, method="xgbTree", trControl=tc, preProc = c("center","scale"))
save(xgb.u,file = paste0(origin,"xgbu.rda"))

rf.b <- train(eventStartYN ~ . , data=dftrainMLb, method="rf", trControl=tc, preProc = c("center","scale"))
save(rf.b,file = paste0(origin,"rfb.rda"))
rf.u <- train(eventStartYN ~ . , data=dftrainMLu, method="rf", trControl=tc, preProc = c("center","scale"))
save(rf.u,file = paste0(origin,"rfu.rda"))


# END PARALLEL PROCESSING
stopCluster(cluster)
registerDoSEQ()


#~~~~~~~~~~~~~~~~~~~~~~~#
#     MODEL TESTING     #
#~~~~~~~~~~~~~~~~~~~~~~~#
dftestML <- read.csv(paste0(origin,"dftestML.csv"), na.strings="NA", stringsAsFactors=TRUE)
library(caret)
library(pROC)


f.mlModelEval <- function(mdel, testdf){
  pred <- cbind(P=predict(mdel, testdf), predict(mdel, testdf, type="prob"))
  AUC <- roc(testdf$eventStartYN, pred$Y)
  cm <- confusionMatrix(pred$P, testdf$eventStartYN, positive = "Y")
  out <- list("AUC" = AUC, "cm" = cm)
  return(out)
}



load(paste0(origin,"lrb.rda"))
load(paste0(origin,"lru.rda"))
load(paste0(origin,"xgbb.rda"))
load(paste0(origin,"xgbu.rda"))
load(paste0(origin,"rfb.rda"))
load(paste0(origin,"rfu.rda"))


res.lr.b <- f.mlModelEval(lr.b, dftestML)
res.lr.u <- f.mlModelEval(lr.u, dftestML)
res.rf.b <- f.mlModelEval(rf.b, dftestML)
res.rf.u <- f.mlModelEval(rf.u, dftestML)
res.xgb.b <- f.mlModelEval(xgb.b, dftestML)
res.xgb.u <- f.mlModelEval(xgb.u, dftestML)

save(res.lr.b,file = paste0(origin,"res.lr.b.rda"))
save(res.lr.u,file = paste0(origin,"res.lr.u.rda"))
save(res.rf.b,file = paste0(origin,"res.rf.b.rda"))
save(res.rf.u,file = paste0(origin,"res.rf.u.rda"))
save(res.xgb.b,file = paste0(origin,"res.xgb.b.rda"))
save(res.xgb.u,file = paste0(origin,"res.xgb.u.rda"))

rm(lr.b, lr.u, rf.b, rf.u, xgb.b, xgb.u, dftestML)


# Sensitivity = correctly predicting starts
# Specificity = correctly predicting non-starts

f.results <- function(mdelStr, res){
  result <- data.frame(Model = mdelStr, 
                       AUC = as.numeric(res$AUC$auc), 
                       Accuracy.Non.Starts = res$cm$byClass["Specificity"],
                       Accuracy.Starts = res$cm$byClass["Sensitivity"],
                       Accuracy.Overall = res$cm$overall["Accuracy"], 
                       Accuracy.Balanced = res$cm$byClass["Balanced Accuracy"],
                       stringsAsFactors = T)
  return(result)
}

results.haz.b <- f.results("Haz Bal", res.haz.b)
results.haz.u <- f.results("Haz Unbal", res.haz.u)
results.lr.b <- f.results("LR Bal", res.lr.b)
results.lr.u <- f.results("LR Unbal", res.lr.u)
results.rf.b <- f.results("RF Bal", res.rf.b)
results.rf.u <- f.results("RF Unbal", res.rf.u)
results.xgb.b <- f.results("XGB Bal", res.xgb.b)
results.xgb.u <- f.results("XGB Unbal", res.xgb.u)


results <- rbind(results.haz.b, results.haz.u, results.lr.u, results.lr.b, results.rf.b, results.rf.u, results.xgb.b, results.xgb.u)
row.names(results) <- NULL
rm(results.haz.b, results.haz.u, results.lr.b, results.lr.u, results.rf.b, results.rf.u, results.xgb.b, results.xgb.u)
write.csv(results, paste0(origin,"results.csv"), row.names = F)


f.plot.results <- function(metric, tbl){
  dotplot( with(tbl,reorder(Model,get(metric))) ~ get(metric),
           data = tbl,
           main = metric,
           xlab = metric,
           ylab = "Model")
}


f.plot.results("AUC", results)
f.plot.results("Accuracy", results)
f.plot.results("Accuracy.Balanced", results)
f.plot.results("Accuracy.Starts", results)
f.plot.results("Accuracy.Non.Starts", results)
results$Average <- rowMeans(results[,c("AUC","Accuracy.Non.Starts","Accuracy.Overall")])
f.plot.results("Average", results)



# EDIT TO HAVE MODEL NAME IN IT FOR PLOT TITLE
f.plotVarImpML <- function(mdel, n, title){
  vi <- varImp(mdel)
  vi <- vi$importance
  vi$Predictor <- row.names(vi)
  vi <- vi[order(-vi$Overall),]
  vi <- vi[1:n,]
  dotplot(with(vi,reorder(Predictor,Overall)) ~ Overall, data = vi, xlab="Importance", main=paste("Relative Variable Importance for",title))
}

f.plotVarImpHAZ <- function(mdel, n, title){
  s <- summary(mdel)
  vi <- data.frame(s$coefficients)
  vi$z.abs <- abs(vi$z)
  vi$Predictor <- row.names(vi)
  vi$Overall = (vi$z.abs-min(vi$z.abs))/(max(vi$z.abs)-min(vi$z.abs))*100
  vi <- vi[order(-vi$Overall),]
  vi <- vi[1:n,]
  dotplot(with(vi,reorder(Predictor,Overall)) ~ Overall, data = vi, xlab="Importance", main=paste("Variable Importance for",title))
}

varImpN <- 20
library(caret)

f.plotVarImpML(xgb.b, varImpN, "Balanced XGBoost")
f.plotVarImpML(lr.b, varImpN, "Balanced Logistic Regression")
f.plotVarImpML(rf.b, varImpN, "Balanced Random Forest")
f.plotVarImpHAZ(haz.u, varImpN, "Balanced Cox-Proportional Hazards Regression")




df <- read.csv(paste0(origin,"df.csv"), na.strings="NA", stringsAsFactors=TRUE)
p <- subset(df, select = c(eventStartYN, daysToFirstSchedClass, coursesScheduled, age))

st.err <- function(x, na.rm=FALSE) {
  if(na.rm==TRUE) x <- na.omit(x)
  sd(x)/sqrt(length(x))
}

f.plotContIVs <- function(df, IV){
  a <- aggregate(eventStartYN ~ get(IV), data = df, mean)
  colnames(a) <- c("IV", "eventStartYN")
  a$se <- aggregate(eventStartYN ~ get(IV), data = p, st.err)[,2]
  a$upper <- a$eventStartYN + a$se
  a$lower <- a$eventStartYN - a$se
  a <- a[complete.cases(a), ]
  xyplot(upper + eventStartYN + lower ~ IV, data = a, type="l", col=c("gray","red"), main=paste("Start Rate by",IV), ylab="Start Rate", xlab=IV)
  #xyplot(eventStartYN ~ IV, data = a, type="l", main=paste("Start Rate by",IV), ylab="Start Rate", xlab=IV)
}

library(lattice)
f.plotContIVs(p, "daysToFirstSchedClass")
f.plotContIVs(p, "coursesScheduled")
f.plotContIVs(p, "age")

with(p, plot(daysToFirstSchedClass,coursesScheduled))

with(p, cor(daysToFirstSchedClass,coursesScheduled))
with(p, cor(daysToFirstSchedClass,age))
with(p, cor(age,coursesScheduled))


max(df$daysToFirstSchedClass)










