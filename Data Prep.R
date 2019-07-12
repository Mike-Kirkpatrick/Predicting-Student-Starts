
origin <- "C:/Users/mkirkpatrick/National University/Precision Institute - Documents/Data Science/Modeling/Student Starts/Saved Objects/"

#~~~~~~~~~~~~~~~~~~~~~#
#     IMPORT DATA     #
#~~~~~~~~~~~~~~~~~~~~~#
rawData <- read.csv(paste0(origin,"rawData.csv"), 
                    na.strings="NULL",
                    stringsAsFactors=TRUE)
colnames(rawData)[colnames(rawData)=="ï..EMPLID"] <- "EMPLID"
df <- subset(rawData, select = c(-EMPLID,-EFFDT_BEGIN,-cip))
DVs <- subset(df, select = c(eventStartYN, daysToEvent))
IVs <- subset(df, select = c(-eventStartYN, -daysToEvent))
IVs <- IVs[ , order(names(IVs))]
df <- data.frame(DVs,IVs)
df <- df[complete.cases(df), ]
rm(DVs, IVs)
rm(rawData)
write.csv(df, paste0(origin,"df.csv"), row.names = F)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     CREATE TRAINING AND TEST DATASETS     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
df <- read.csv(paste0(origin,"df.csv"), na.strings="NA", stringsAsFactors=TRUE)

# FOR HAZARD MODELS
library(caret)
set.seed(1)
inTraining <- createDataPartition(df$eventStartYN, p = .70, list = FALSE)
dftrain <- df[ inTraining,]
dftest  <- df[-inTraining,]
rm(inTraining)

write.csv(dftrain, paste0(origin,"dftrainHAZunbalanced.csv"), row.names = F)
write.csv(dftest, paste0(origin,"dftestHAZ.csv"), row.names = F)

# BALANCE THE CLASSES IN HAZ TRAINING DATASET
dftrainHAZ <- read.csv(paste0(origin,"dftrainHAZ.csv"), na.strings="NA", stringsAsFactors=TRUE)
row.names(dftrainHAZ) <- NULL
n <- dftrainHAZ[dftrainHAZ$eventStartYN==0,]
y <- dftrainHAZ[dftrainHAZ$eventStartYN==1,]
set.seed(1)
y <- y[sample(1:nrow(y),(table(dftrainHAZ$eventStartYN))[1], replace = F),]
dftrainB <- rbind(n,y)
set.seed(1)
dftrainB <- dftrainB[sample(nrow(dftrainB)),] #shuffle rows
rm(n,y)

write.csv(dftrainB, paste0(origin,"dftrainHAZbalanced.csv"), row.names = F)
rm(dftrainB)



# FOR ML MODELS
dftrain <- subset(dftrain, select = -daysToEvent)
dftest <- subset(dftest, select = -daysToEvent)

dftrain$eventStartYN[dftrain$eventStartYN==0] <- "N"
dftest$eventStartYN[dftest$eventStartYN==0] <- "N"
dftrain$eventStartYN[dftrain$eventStartYN==1] <- "Y"
dftest$eventStartYN[dftest$eventStartYN==1] <- "Y"
dftrain$eventStartYN <- as.factor(dftrain$eventStartYN)
dftest$eventStartYN <- as.factor(dftest$eventStartYN)

write.csv(dftest, paste0(origin,"dftestML.csv"), row.names = F)
write.csv(dftrain, paste0(origin,"dftrainMLunbalanced.csv"), row.names = F)



# BALANCE THE CLASSES IN HAZ TRAINING DATASET
row.names(dftrain) <- NULL
n <- dftrain[dftrain$eventStartYN=="N",]
y <- dftrain[dftrain$eventStartYN=="Y",]
set.seed(1)
y <- y[sample(1:nrow(y),(table(dftrain$eventStartYN))[1], replace = F),]
dftrainB <- rbind(n,y)
set.seed(1)
dftrainB <- dftrainB[sample(nrow(dftrainB)),] #shuffle rows
rm(n,y)

write.csv(dftrainB, paste0(origin,"dftrainMLbalanced.csv"), row.names = F)


