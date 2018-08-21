#UČENJE MODELA

library(caret)
library(randomForest)
library(mxnet)
library(clue)

#cena napake 
costPenalty <- function(data, levels, ...){
  falseNegative <- sum(data$pred==levels[1] & data$obs == levels[2])
  actualPositive <- sum(data$obs==levels[2])
  fnr <- falseNegative / actualPositive
  falsePositive <- sum(data$pred==levels[2] & data$obs == levels[1])
  actualNegative <- sum(data$obs==levels[1])
  fpr <- falsePositive / actualNegative
  
  cost = fpr + 0.5*fnr
  names(cost) <- 'cost'
  cost
}

#trainControl 

tc <- trainControl(method = "cv",number = 10,savePredictions = TRUE,classProbs = FALSE,summaryFunction = costPenalty,verboseIter = TRUE)

#klasična funkcija izgube za klasifikacijo
calAcc <- function(model, testData, testLabel){
  sum(predict(model, testData) == testLabel)/nrow(testData)
}


bestModel <- function(trainX, trainY){
  #Funkcij bestModel prejme učno množico ter vrne najboljši model
  
  costs <- vector()
  acc <- vector()#za shranjevanje natancnosti modelov in vrednosti cost
  models <- list() #shranjevanje modelov
  
  #1. metoda najbližjih sosedov
  modelKNN <- train(x=trainX, y=trainY,
                    method = "knn",
                    metric = "cost",
                    maximize = "false",
                    trControl = tc,
                    tuneGrid = data.frame(k=2:10))
  #izracunamo natancnost in shranimo model
  acc[[1]] <- calAcc(modelKNN,trainX,trainY)
  costs[[1]]<- min(modelKNN$results$cost)
  models[[1]] <- modelKNN
  
  #2. logistična regresija
  modelLR <- train(x=trainX, y=trainY, method='glm',trControl = tc)
  
  acc[[2]] <- calAcc(modelLR,trainX,trainY)
  costs[[2]] <- modelLR$results$cost
  models[[2]] <- modelLR
  
  #3. odločitveno drevo
  rpartModel <- train(x= trainX,
                      y = trainY,
                      method='rpart2', 
                      trControl=tc)
  
  acc[[3]] <- calAcc(rpartModel,trainX,trainY)
  costs[[3]] <- min(rpartModel$results$cost)
  models[[3]] <- rpartModel
  
  #4. metoda podpornih vektorjev z linearno mejo
  
  for(i in -3:3){
    j = i+4
    svModel <- train(x=trainX, y=trainY,
                     method = 'svmLinear',
                     metric = "cost",
                     maximize = "false",
                     scale = FALSE,
                     trControl = tc,
                     tuneGrid = data.frame(C=exp(i)))

    acc[[3+j]]<- calAcc(svModel,trainX,trainY)
    costs[[3+j]] <- svModel$results$cost
    models[[3+j]] <-svModel
  }
  
  #5. Random forrest
  
  rfModel<- train(x=trainX,y=trainY,
              method="rf", metric="cost", ntree=200,
              tuneGrid=expand.grid(.mtry=c(10)), trControl=tc)
  acc[[11]] <-calAcc(rfModel,trainX,trainY)
  costs[[11]] <- min(rfModel$results$cost)
  models[[11]] <- rfModel
  
  
  #6. metoda podpornih vektorjev (radial) in polinomsko jedro
  Cs <- exp(-3:3)

  # radialno jedro:

  sigmas <- exp(-2:2)
  svmRadial <- train(x=trainX, y=trainY,
                      method = 'svmRadial',
                      metric = "cost",
                      maximize = "false",
                      scale = FALSE,
                      trControl = tc,
                      tuneGrid = expand.grid(C=Cs, sigma=sigmas))

  acc[[12]] <-calAcc(svmRadial,trainX,trainY)
  costs[[12]] <- min(svmRadial$results$cost)
  models[[12]] <- svmRadial

  
  #vrne najboljši model na dani množici
  models[[which.min(costs)]]
  
}

trainX <- trainData[,1:16]
trainY <- trainData$Lodobren


#izberemo najboljši model

finalModel <- bestModel(trainX,trainY)

#napoved
napoved <- predict(finalModel,testData)
levels(napoved)<-c("NE","DA")

#izvozimo
write.csv(napoved, quote=FALSE,row.names=FALSE,file="napovedi.csv",fileEncoding = "UTF-8")



