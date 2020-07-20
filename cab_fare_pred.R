#Clean the environment
rm(list = ls())

# Set working directory
setwd("C:\\Users\\M1053735\\Pictures\\Cab_Fare_pred")

# Load required Libraries for analysis  ----------------------------------
x = c("ggplot2", "corrgram", "usdm", "caret", "DMwR", "rpart", "randomForest",'xgboost','moments','car','DataCombine','rsq','geosphere')

#load Packages
lapply(x, require, character.only = TRUE)
rm(x)
train = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test = read.csv("test.csv")
# Structure of data
str(train)
str(test)
summary(train)
summary(test)
head(train,5)
head(test,5)

# Changing the data types of variables
train$fare_amount = as.numeric(as.character(train$fare_amount))
train$passenger_count = as.integer(train$passenger_count)
train$pickup_datetime = as.POSIXct(train$pickup_datetime,format="%Y-%m-%d %H:%M:%S",tz="UTC")
test$pickup_datetime = as.POSIXct(test$pickup_datetime,format="%Y-%m-%d %H:%M:%S",tz="UTC")

# Data Cleaning
# fare amount cannot be less than one and considering 453 as max because only 2 observations are there greater than 453 
# passenger count range 1-6
# Latitudes range from -90 to 90, and longitudes range from -180 to 180.
train=subset(train, !(train$fare_amount<1))
train=subset(train,!(train$fare_amount>453))
train=subset(train,!(train$passenger_count<1))
train=subset(train,!(train$passenger_count>6))
sum(train$pickup_latitude>90) #..1
sum(train$pickup_latitude < (-90))#..0
sum(train$pickup_longitude>180) #..0
sum(train$pickup_longitude<(-180)) #..0

sum(train$dropoff_latitude>90) #..0
sum(train$dropoff_latitude < (-90))#..0
sum(train$dropoff_longitude>180) #..0
sum(train$dropoff_longitude<(-180)) #..0

train=subset(train,!(train$pickup_latitude>90))
#Zero degrees latitude is the line designating the Equator and divides the Earth into two equal hemispheres (north and south). Zero degrees longitude is an imaginary line known as the Prime Meridian.
for(i in list('pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')){
  print(paste0(i," equal to 0 = ",sum(train[[i]]==0)))
}

#removing rows
for(i in list('pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude')){
  train=subset(train,!(train[[i]])==0)
}

#Eliminate rows where the pickup and drop location points are same
train=subset(train,!(train$pickup_longitude==train$dropoff_longitude & train$pickup_latitude==train$dropoff_latitude))

# checking for missing values.
sum(is.na(train))
sum(is.na(test))
train=na.omit(train)#remove missing values

#calculate distance for train
train$dist = distHaversine(cbind(train$pickup_longitude,train$pickup_latitude),cbind(train$dropoff_longitude,train$dropoff_latitude))
#the output is in metres, Change it to kms
train$dist=as.numeric(train$dist)/1000

#calculate distance for test
test$dist = distHaversine(cbind(test$pickup_longitude,test$pickup_latitude),cbind(test$dropoff_longitude,test$dropoff_latitude))
#the output is in metres, Change it to kms
test$dist=as.numeric(test$dist)/1000

#Now we know the starting date in our dataset. we will take a date 1 month before our starting date and take the difference for each row and create a new variable and then we will take that new variable into consideration.
sort(train$pickup_datetime,decreasing = FALSE)
reference_date = as.POSIXct("2008-12-01 01:31:49",format="%Y-%m-%d %H:%M:%S",tz="UTC")
#reset rownames
row.names(train) <- NULL

train$elapsed_sec = difftime(train$pickup_datetime,reference_date,units = "secs")
test$elapsed_sec = difftime(test$pickup_datetime,reference_date,units = "secs")

train$elapsed_sec = as.numeric(as.character(train$elapsed_sec))
test$elapsed_sec = as.numeric(as.character(test$elapsed_sec))



#drop the columns which are not needed
droplist = c('pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','elapsed_time')

train=train[,!colnames(train) %in% droplist]
test=test[,!colnames(test) %in% droplist]

continuous_variables = colnames(train)


##Outliers Analysis
for(i in 1:length(continuous_variables)){
  assign(paste0("gn",i), ggplot(data = train, aes_string(x = "fare_amount", y = continuous_variables[i]))+
           stat_boxplot(geom = "errorbar",width = 0.5)+
           geom_boxplot(outlier.colour = 'red',fill='grey',outlier.shape = 18,outlier.size = 4)+
           labs(y=continuous_variables[i],x='cnt')+
           ggtitle(paste("Box plot of",continuous_variables[i])))
}


gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)

df_train = train

##Treating outliers
outliers_vars = c('fare_amount','dist', 'elapsed_sec')
for(i in outliers_vars){
  temp = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  df_train[,i][df_train[,i] %in% temp] = NA
}

# check for missing values
apply(df_train,2, function(x){ sum(is.na(x))})

#we found 1342 outliers in fare_amount and 1334 in dist
df_train = knnImputation(data = df_train, k = 3)

##As we have only 3 variables we will not do feature selection

#function to vizualize the continuity of continuous variables
continuos_Vars_display <- function(cntnus){
  ggplot(df_train) +
    geom_histogram(aes(x = cntnus, y = ..density..),fill='green',colour='black') +
    geom_density(aes(x = cntnus, y = ..density..)) 
  
}

continuos_Vars_display(df_train$fare_amount)
continuos_Vars_display(df_train$dist)
continuos_Vars_display(df_train$elapsed_sec)

#check the skewness
for(i in colnames(df_train)){
  print(paste0("skewness of ",i," = ",skewness(df_train[[i]])))
}

#Let's Normalize the data
df_train$elapsed_sec = (df_train$elapsed_sec - min(df_train$elapsed_sec)) / (max(df_train$elapsed_sec) - min(df_train$elapsed_sec))
test$elapsed_sec = (test$elapsed_sec - min(test$elapsed_sec)) / (max(test$elapsed_sec) - min(test$elapsed_sec))

#Let's perform log transformation to reduce skewness
df_train$dist = log1p(df_train$dist)
df_train$fare_amount = log1p(df_train$fare_amount)
test$dist = log1p(test$dist)


####Splitting the data
set.seed(12345)
train_index = createDataPartition(df_train$fare_amount, p = 0.8, list=FALSE)
new_train = df_train[train_index,]
new_test = df_train[-train_index,]

rmExcept(c("new_train",'new_test','test'))

###Model development
# Linear Regression
linear_regressor = lm(fare_amount ~.,data = new_train)
summary(linear_regressor)
pred = predict(linear_regressor,new_test[,-1])
regr.eval(test[,1],preds = pred)

#   mae         mse          rmse      mape 
#1.2412741  2.0523051   1.4325868  0.9565344 

#calculate R-Squared value
rsq(fitObj = linear_regressor,adj = TRUE,data = new_train)
#0.732751

### Decision Tree
tree = rpart(fare_amount ~ ., data=new_train, method = "anova")
summary(tree)

pred_dt = predict(tree, new_test[,-1])

regr.eval(new_test[,1],preds = pred_dt)

#     mae        mse            rmse       mape 
#0.16549514   0.04579052   0.21398719   0.07753618

rss_dt = sum((pred_dt - new_test$fare_amount) ^ 2)
tss_dt = sum((new_test$fare_amount - mean(new_test$fare_amount)) ^ 2)
rsq_dt = 1 - rss_dt/tss_dt
#0.7000909

##Random Forest
rf_model = randomForest(fare_amount ~.,data=new_train, importance = TRUE, ntree=500)
summary(rf_model)

pred_rm = predict(rf_model,new_test[-1])

regr.eval(new_test[,1],preds = pred_rm)
#    mae            mse        rmse        mape 
# 0.16099518  0.04287104   0.20705322   0.07593300  

# calculate R-Square value
rss_rf = sum((pred_rm - new_test$fare_amount) ^ 2)
tss_rf = sum((new_test$fare_amount - mean(new_test$fare_amount)) ^ 2)
rsq_rf = 1 - rss_rf/tss_rf
#0.7192123

##XgBoost

train_data_matrix = as.matrix(sapply(new_train[-1],as.numeric))
test_data_matrix = as.matrix(sapply(new_test[-1],as.numeric))

xgb = xgboost(data = train_data_matrix,label = new_train$fare_amount, nrounds = 13,verbose = TRUE)

pred_xgb = predict(xgb,test_data_matrix)

regr.eval(new_test[,1],preds = pred_xgb)
#  mae           mse          rmse       mape 
#0.14578432   0.03761938  0.19395717   0.06733712 


#calculate R-Square value
rss_xgb = sum((pred_xgb - new_test$fare_amount) ^ 2)
tss_xgb = sum((new_test$fare_amount - mean(new_test$fare_amount)) ^ 2)
rsq_xgb = 1 - rss_xgb/tss_xgb
#0.7536085


# from the above models we can conclude that XGBOOST Model is the best fit for this problem.

#Prediction on our test data
Test_matrix = as.matrix(sapply(test,as.numeric))
test_pred = predict(xgb,Test_matrix)
write.csv(test_pred,"Predicted_Data.csv",row.names = F)