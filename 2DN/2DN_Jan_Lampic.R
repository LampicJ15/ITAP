library(caret)
library(dplyr)
library(rpart)
library(mxnet)
library(randomForest)
library(clue)

#uvozimo podatke
library(readxl)
data <- read_excel("~/GitHub/ITAP/2DN/data.xlsx")

summary(data)
#UREDIMO PODATKE ====================================================================================
#imena vrstic
row.names(data)<- data$TextID 
#uredimo napovedno spremenljivko
data$Label <- as.factor(data$Label)

#zbrisem stolpec url in TextID
data$URL <- NULL
data$NNP <- NULL #same 0
data$WRB <- NULL #same nule

#testData 
#testData za preverjanje končne funkcije
test <- data[sample(nrow(data), 50), ]

rownames(test)<-test$TextID
data$TextID <- NULL

#====================================================================================================
#uredimo podatke
#funkcija editData uredu podatke in jih normaliira če je norm=TRUE, sicer jih standardizira
#vrnse seznam data.frame urejenih podatkov, maximume stolpcev oz. povprečja in minimume oz. standardne odklone stolpcev
editData <- function(inputData, norm){
  Max <- vector()
  Min <- vector()
  for (i in 1:ncol(inputData))
    
    if (is.numeric(inputData[[i]])){
      inputData[[i]]<-as.numeric(inputData[[i]])
      if (norm){
        Max[i] <- max(na.omit(inputData[i]))
        Min[i] <- min(na.omit(inputData[i]))
        inputData[[i]]<- (inputData[[i]] - Min[i])/(Max[i] - Min[i])
      }else{
        Max[i] <- mean(na.omit(inputData[[i]]))
        Min[i]<- sd(na.omit(inputData[[i]]))
        #standardiziramo
        inputData[[i]] <- (inputData[[i]]-Max[i])/Min[i]
      }
    } else{
      inputData[[i]] <- as.factor(inputData[[i]])
    }
  return(list(inputData,Max,Min))
}

#funkcija editTestData nastavi prave tipe stolpcev in jih normalizira, če je normalizacija na TRUE oz standardizira, če je na FALSE
#vrne normalizirane ali standardizirane stolpce in z vrednostmi iz učne množice

editTestData <- function(inputData, norm,Max,Min){
  
  for (i in 1:ncol(inputData))
    
    if (is.numeric(inputData[[i]])){
      inputData[[i]]<-as.numeric(inputData[[i]])
      if (norm){
        #normaliziramo
        inputData[[i]]<- (inputData[[i]] - Min[i])/(Max[i] - Min[i])
      }else{
        #standardiziramo
        inputData[[i]] <- (inputData[i]-Max[i])/Min[i]
      }
    } else{
      inputData[[i]] <- as.factor(inputData[[i]])
    }
  return(inputData)
}

sez <- editData(data,TRUE)
data <- sez[[1]]
Max <- sez[[2]][2:58]
Min <- sez[[3]][2:58]



#====================================================================================================
#PARAMETRI 

#določimo vrednost parametra p za razbitje učne množice
p = 0.8
#traincontrol
tc <- trainControl(method='none', savePredictions = TRUE)
#====================================================================================================

#funkcija napake
calAcc <- function(model, testData, testLabel){
  sum(predict(model, testData) == testLabel)/nrow(testData)
}


#====================================================================================================
#izbira vrednosti parametra k
#to naredimo s t.i. elbow method(komolčna metoda)
set.seed(123)
# Izračunajmo vsoto razdalj znotraj gruč za k = 2 do k = 15.
k.max <- 15
vsotaRazdalj <- sapply(1:k.max, 
              function(k){kmeans(data[,-1], k, nstart=50,iter.max = 15 )$tot.withinss})
vsotaRazdalj
plot(1:k.max, vsotaRazdalj,
     type="b", pch = 19, frame = FALSE, 
     xlab="Število gruč k",
     ylab="Vsota kvadrata razdalj znotraj gruč")

#graf ima komolec pri k=3
k <- 3
#====================================================================================================

bestModel <- function(trainData, testData){
  #Funkcij bestModel prejme učno in testno množico ter vrne najboljši model
  testLabel <- testData$Label
  testData <- testData[,names(testData)!="Label"]
  
  
  acc <- vector() #za shranjevanje natancnosti modelov
  models <- list() #shranjevanje modelov
  
  #1. Random forest
  rfModel<- train(Label~.,data=trainData,
                  method="rf",ntree=200,
                  tuneGrid=expand.grid(.mtry=c(10)), trControl=tc)
  
  acc[[1]]<- calAcc(rfModel, testData,testLabel)
  models[[1]] <- rfModel

  
  #2. logistična regresija
  modelLR <- train(Label~., data=trainData, method='glm',trControl = tc)
  
  acc[[2]] <- calAcc(modelLR, testData,testLabel)
  models[[2]] <- modelLR
  
  #3. odločitveno drevo
  rpartModel <- train(x= trainData[,names(trainData)!= "Label"],
                      y = trainData$Label,
                      method='rpart2', 
                      trControl=tc)
  
  acc[[3]] <- calAcc(rpartModel, testData,testLabel)
  models[[3]] <- rpartModel
  
  #4. metoda podpornih vektorjev z linearno mejo
  
  for(i in -3:3){
    j = i+4
    svModel <- train(Label~., data=trainData,
                       method = 'svmLinear',
                       scale = FALSE,
                       trControl = trainControl(method='none'),
                       tuneGrid = data.frame(C=exp(i)))
  
    acc[[3+j]] <- calAcc(svModel, testData,testLabel)
    models[[3+j]] <-svModel
  }
  
  
  #vrne najboljši model na dani množici
  models[[which.max(acc)]]

}

finalPredict <- function(inputData, k){
  #napove razred za data frame inpuData, ki je enake oblike kot 'data', vendar brez razreda 'Label'
  row.names(inputData)<- inputData$TextID 
  
  #output vektor (napovedi)
  output <- data.frame(Label=rep(0,nrow(inputData)))
  row.names(output)<-inputData$TextID

  #zbrisem stolpec url in TextID
  inputData$URL <- NULL
  inputData$TextID <- NULL
  inputData$NNP <- NULL #same 0
  inputData$WRB <- NULL #same 0
  
  #podatke normaliziramo z vrednostmi iz učne množice
  inputData <- editTestData(inputData,TRUE,Max,Min)
  
  #razvrstimo  k gruč
  grupiranje <- kmeans(data[,-1],k)
  data$cluster <- grupiranje$cluster
  
  #vsakemu podatku pripišemo najbližji center
  inputData$cluster <- cl_predict(grupiranje, newdata=inputData)
  
  #zberemo najboljse modele za k gruč
  kBestModels <- list()
  
  for(i in 1:k){
    print(paste(c("Učim se model na gruči",i), collapse=" "))
    
    tmpData <- data[data$cluster == i,names(data)!="cluster"]
    
    #razdelimo tmpData na učno in testno množico
    trainIndex <- createDataPartition(tmpData$Label, p=p, list=FALSE);
    tmpTrainData <- tmpData[trainIndex,];
    tmpTestData <- tmpData[-trainIndex,];
    
    kBestModels[[i]] <- bestModel(tmpTrainData,tmpTestData)

  }
  
  #napovemo vrednosti za inputData
  
  for (i in 1:k){
    if (sum(inputData$cluster==i)!= 0){
    output$Label[inputData$cluster == i] <- predict(kBestModels[[i]], inputData[inputData$cluster==i, names(inputData)!='cluster'])
    }
  }
  output$Label <- factor(output$Label,levels = c(2,1),labels = c("subjective","objective"))
  output
}

#zagon končne funkcije
#primer: 50 random vrstic iz podatkov
input <- test
input$Label <- NULL

results <- finalPredict(input,k=3)
