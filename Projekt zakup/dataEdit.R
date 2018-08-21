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

#polno nadzorovano učenje - urejamo vse podatke tudi iz testne množice
data <- rbind(trainData, testData)
#uredimo tipe stoplcev 

#funkcija editData nastavi prave tipe stolpcev in jih normalizira, če je normalizacija na TRUE oz standardizira, če je na FALSE


editData <- function(inputData, norm){
  for (i in 1:ncol(inputData))
    
    if (is.numeric(inputData[[i]])){
      inputData[[i]]<-as.numeric(inputData[[i]])
      if (norm){
        #normaliziramo
        Max <- max(na.omit(inputData[[i]]))
        Min <- min(na.omit(inputData[[i]]))
        inputData[[i]]<- (inputData[[i]] - Min)/(Max- Min)
      }else{
        #standardiziramo
        inputData[[i]] <- scale(inputData[[i]])
      }
    } else{
      inputData[[i]] <- as.factor(inputData[[i]])
    }
  return(inputData)
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
        inputData[[i]] <- scale(inputData[[i]])
      }
    } else{
      inputData[[i]] <- as.factor(inputData[[i]])
    }
  return(inputData)
}

#uredimo podatke
data <-editData(data,TRUE)


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

data[,2:31] <- replaceNA(data[,2:31])

#dodamo boolove spremenljivke zato, da lahko izračunamo variance in koovariance
addDummy <- function(inputData){
  inputData <- data.table(dummy.data.frame(inputData))
  return(inputData)
}

data <- addDummy(data)
# odstranimo nepotrebne dummy stolpce (funkcija je naredila nekaj stolpcev prevec)
data <- data[,-c("LodobrenNE", "VznamkaALFA", "Scrna_listaNE",
                           "Dcrna_listaNE","Strajanje_zaposlitveUPOKOJENEC", "SpostaBA-7", "SdrzavljanstvoALBANIA",                
                           "Sdrzavljanstvo_euNE", "SDtipSVOBOD. POKL., TUJINA, D", "SDblokada_racunaNE", "SDinsolventnostNE",                      
                           "SDcrna_listaNE")]
data$Lodobren <- as.factor(data$LodobrenDA)
data$Lodobren[data$LodobrenNA == 1] <- NA
data$LodobrenNA <- NULL
data$LodobrenDA <- NULL


#izbira pomembnih napovednih spremenljivk

#zagotovimo najprej da se rezultati lahko ponovijo
set.seed(123)

#odstranimo spremenljivke z nizko varianco (Near Zero-Variance Predictors)
nzv <- nearZeroVar(data[,1:106], names = TRUE) 
#imamo kar 70 spremenljivk z nizko varianco
data[,nzv] <- NULL

#izračunajmo kovariančno matriko
numericData <- data[,c(1:36)]

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
highlyCorCol <- colnames(numericData)[highlyCorrelated]

#odstranimo visoko korelirane spremenljivke
data[,highlyCorrelated]<-NULL

#ločimo podatke
trainData <- data[!is.na(data$Lodobren)]
testData <- data[is.na(data$Lodobren)]

#Rangirajmo napovedne spremenljivke, glede na pomembnost
control <- trainControl(method="repeatedcv", number=10, repeats=3, verboseIter = FALSE)
#naučimo
model <- train(Lodobren~., data=trainData, method="lvq",  trControl=control)
# ocenimo pomembnost spremenljivk
importance <- varImp(model, scale=FALSE)
print(importance)
#plot importance
plot(importance)

 
#RFE metoda za določanje pomembnih napovednih spremenljivk
control2 <- rfeControl(functions = rfFuncs, method = 'cv', number = 10 )
results <- rfe(x=trainData[,1:53], y=trainData$Lodobren, sizes = c(10:53), rfeControl = control2)
print(results)
# seznam pomembnih napovednih spremenljivk
napovedne <- predictors(results)
napovedne[length(napovedne)+1]<-'Lodobren'
# naredimo graf
plot(results, type=c("g", "o"))

#odstranimo nepomembne spremenljivke
trainData <- subset(trainData, select=napovedne)


#preoblikujmo še razmerje

# razmerje "DA": 60%, "NE": 40%
# preoblikuj na "DA": 48%, "NE": 52%
trainData <- SMOTE(Lodobren~., trainData, perc.over = 30, k = 5, perc.under = 400)





