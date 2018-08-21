library(readr)
library(caret)
library(dplyr)
library(rpart.plot)
library(rpart)

trainData <- read.csv("ucni_podatki.csv",header = FALSE)
names(trainData) <- c("index","X1","X2","X3","X4","X5","X6")

Class <- read.csv("ucni_razredi.csv", header = FALSE)
names(Class) <- c("index","Y")
levels(Class$Y)[levels(Class$Y) == ""] <- NA



Data <- read.csv("testni_podatki.csv", header = FALSE)
names(testData) <- c("index","X1","X2","X3","X4","X5","X6")


# združimo učne razrede z učnimi podatki

trainData <- inner_join(trainData, Class, by = "index")
trainData <- trainData[,c(-1)]


#uredimo učno podatkovno množico

#preverimo tipe stoplcev
#stolpca X1 in X3 sta factor, jih spremenimo v numeric, ne numericne vrednosti pa v NA
trainData$X1 <- as.numeric(levels(trainData$X1))[trainData$X1]
trainData$X3 <- as.numeric(levels(trainData$X3))[trainData$X3]

#odstranimo vse neznane vrednosti (NA)
trainData <- na.omit(trainData)

#spremenimo vrednosti v Y D in N v 1 in 0
trainData$Y <- as.integer(trainData$Y)
trainData$Y[trainData$Y == 2] <- 0
trainData$Y <- as.factor(trainData$Y)

levels(trainData$Y)
# levels(trainData$Y)[1] = 0

######################################################
#Treniranje modela

F_beta <- function(data, levels, ...) {
  beta <- 2
  TP <- sum(data$obs==levels[2] & data$pred==levels[2]) #true positives
  FP <- sum(data$obs==levels[1] & data$pred==levels[2]) #false positives
  FN <- sum(data$obs==levels[2] & data$pred==levels[1]) #false negatives
  P <- TP + FN
  PPV <- TP / (TP + FP)
  TPR <- TP / P
  f_beta <- (1+beta*beta)*PPV*TPR/(PPV*beta*beta + TPR)
  names(f_beta) <- 'F_beta'
  return(f_beta)
}



##Metoda najbližjih sosedov
cvtc <- trainControl(method='cv', number = 10,savePredictions = TRUE, summaryFunction = F_beta)
dataGrid <- data.frame(k=1:30) #da bomo pognali metodo za vrednosti od 1 do 30
knnModel <- train(Y ~ ., data=trainData, method="knn",metric ='F_beta' ,tuneGrid=dataGrid, trControl =cvtc)

#Optimalni k pri metodi najbližjih sosedov je k=7 vrednost F_beta 0.6410491

##Logistična regresija

# Ker testiramo več modelov na istem razbitju:
cvtc <- trainControl(method='cv', index = createFolds(trainData$Y, k=10, returnTrain=TRUE), 
                     summaryFunction = F_beta)

errors <- data.frame(deg=1:5, err=rep(-1,5), sd=rep(-1,5))

for(k in 1:5){
  print(k)
  tmpData <- data.frame(poly(as.matrix(trainData[, 1:6]), degree=k, raw=TRUE), Y=trainData$Y)
  tmpModel <- train(Y~., data=tmpData, method='glm', trControl=cvtc)
  
  # Izračunane napake
  errors$err[k] <- tmpModel$results$F_beta
  errors$sd[k] <- tmpModel$results$F_betaSD
 
}
errors


#najboljši model je pri parametru k = 1, F_beta = 0.68

logModel <- train(Y~., data=trainData, method='glm', trControl=cvtc)


##Odločitveno drevo

rpartModel <- train(Y~., data=trainData,
                  method='rpart2', 
                  trControl=trainControl(method='cv', number=10,summaryFunction = F_beta))

#F_beta je tu enak 0.64

myDataX <- trainData[names(trainData)!='Y']
myDataY <- trainData$Y
myModelrpartLike <- train(myDataX, myDataY, 
                          method='rpart', 
                          trControl=trainControl(method='cv', number=10, summaryFunction = F_beta),
                          tuneGrid = data.frame(cp=0),
                          control=rpart.control(maxdepth = 30, minsplit=1, xval=0))

#F_beta je tu enak 0.61372

###Napovedvoanje s pomočjo najboljšega modela

Data <- read.csv("testni_podatki.csv", header = FALSE)
names(Data) <- c("index","X1","X2","X3","X4","X5","X6")

#tezave v X1
Data$X1 <- as.numeric(as.character(Data$X1))
Data$X1[20] <-  8.38371346378874 * 10^-5

#odstranimo vse neznane vrednosti (NA)
Data <- na.omit(Data)

#napovemo z logističnim modelom
Data$pred <- predict(logModel, Data[,c(2:7)])
Data$pred <-ifelse(Data$pred==0, 'N','D')

resitev <- Data[c(1,8)]
write.table(resitev, file = "resitev.csv",row.names = FALSE, col.names = FALSE, sep = ',', quote = FALSE)
