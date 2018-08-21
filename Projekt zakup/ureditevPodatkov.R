#Knjiznice
library(readr)
library(dplyr)
library(data.table)
library(randomForest)
library(dummies)
library(DMwR)
library(caret)
library(mlbench)
library(corrplot)
library(plyr)


#UVOZ PODATKOV
trainData<- read_csv("C:/Users/Uporabnik/Dropbox/Faks/3.letnik/ITAP/Zakup/leasing-train.csv")
testData <- read_csv("C:/Users/Uporabnik/Dropbox/Faks/3.letnik/ITAP/Zakup/leasing-test.csv")

#uredimo tipe stoplcev 

#funkcija editData nastavi prave tipe stolpcev in jih normalizira, če je normalizacija na TRUE oz standardizira, če je na FALSE


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

#urejanje podatkov
sez <- editData(trainData, TRUE)
trainData <- sez[[1]]
Max <- sez[[2]]
Min <- sez[[3]]
testData <-editTestData(testData,TRUE,Max,Min)


#nadomeščanje manjkajočih podatkov

#stolpcema Splacilni_indeks in Szapadlo_neplacano manjkajo podatki
replaceNA <- function(inputData){
  
  for (i in 1:ncol(inputData)){
    if (sum(is.na(inputData[[i]]))>0){
      inputData[[i]] <- na.roughfix(inputData[[i]])
    }
    
  }
  return(inputData)
}

trainData <- replaceNA(trainData)
testData[,2:31]<-replaceNA(testData[,2:31])

#dodamo boolove spremenljivke zato, da lahko izračunamo variance in koovariance
addDummy <- function(inputData){
  inputData <- data.table(dummy.data.frame(inputData))
  return(inputData)
}

trainData <- addDummy(trainData)
testData <- addDummy(testData)

# odstranimo nepotrebne dummy stolpce (funkcija je naredila nekaj stolpcev prevec)
trainData <- trainData[,-c("LodobrenNE", "VznamkaALFA", "Scrna_listaNE",
                           "Dcrna_listaNE","Strajanje_zaposlitveUPOKOJENEC", "SpostaBA-7", "SdrzavljanstvoALBANIA",                
                           "Sdrzavljanstvo_euNE", "SDtipSVOBOD. POKL., TUJINA, D", "SDblokada_racunaNE", "SDinsolventnostNE",                      
                           "SDcrna_listaNE")]
trainData$Lodobren <- as.factor(trainData$LodobrenDA)
trainData$LodobrenDA <- NULL


#izbira pomembnih napovednih spremenljivk

#zagotovimo najprej da se rezultati lahko ponovijo
set.seed(123)

#odstranimo spremenljivke z nizko varianco (Near Zero-Variance Predictors)
nzv <- nearZeroVar(trainData[,1:103], names = TRUE) 

trainData[,nzv] <- NULL

#izračunajmo kovariančno matriko
numericData <- trainData[,c(1:35)]

corMat <- cor(numericData)

#pregled kovariančnee matrike
print(corMat)
summary(corMat[upper.tri(corMat)])

#graf korelacijskih vrednosti
corrplot(corMat, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))

#poiščemo spremenljivke, ki so močno korelirane
highlyCorrelated <- findCorrelation(corMat, cutoff=0.85, names = TRUE)

#indeksi močno koreliranih
print(highlyCorrelated)

#odstranimo visoko korelirane spremenljivke
trainData[,highlyCorrelated]<-NULL


#Rangirajmo napovedne spremenljivke, glede na pomembnost
control <- trainControl(method="repeatedcv", number=10, repeats=3, verboseIter = TRUE)
#naučimo
model <- train(Lodobren~., data=trainData, method="lvq",  trControl=control)
# ocenimo pomembnost spremenljivk
importance <- varImp(model, scale=FALSE)
print(importance)
#plot importance
plot(importance)

 
#RFE metoda za določanje pomembnih napovednih spremenljivk
control2 <- rfeControl(functions = rfFuncs, method = 'cv', number = 10 )
results <- rfe(x=trainData[,1:34], y=trainData$Lodobren, sizes = c(10:34), rfeControl = control2)
print(results)
# seznam pomembnih napovednih spremenljivk
napovedne <- predictors(results)
napovedne[length(napovedne)+1]<-'Lodobren'
# naredimo graf
plot(results, type=c("g", "o"))

#odstranimo nepomembne spremenljivke
trainData <- subset(trainData, select=napovedne)
testData <- subset(testData, select=napovedne[1:16])

#preoblikujmo še razmerje
trainData <- SMOTE(Lodobren~., trainData, perc.over = 30, k = 5, perc.under = 400)





