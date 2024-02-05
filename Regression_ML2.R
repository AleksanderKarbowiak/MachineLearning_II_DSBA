library(tidyverse)
library(tree)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(here)
library(corrplot)
library(neuralnet)
library(gbm)
library(boot) #cross-validation
## Loading UBL to balance the unbalanced target
library(UBL)

setwd(here())
getwd()
house_data <- read.csv(file="r1.csv",header=TRUE,sep=",")
house_data_init <- house_data

str(house_data)
colSums(is.na(house_data)) %>% 
  sort()
head(house_data)

table(house_data$waterfront)
table(house_data$bathrooms)
table(house_data$bedrooms)
table(house_data$floors)

#duplicates
house_data_reduced <- house_data[,!(names(house_data) %in% c("id"))]
house_data_reduced[duplicated(house_data_reduced)]


#distribution of price var
a <- ggplot(house_data, aes(x = price))
a + geom_density() +
  geom_vline(aes(xintercept = mean(price)), 
             linetype = "dashed", size = 0.6, colour="red") + geom_vline(aes(xintercept = median(price)), 
                                                           linetype = "dashed", size = 0.6, colour="green") + scale_x_continuous() + theme_minimal()#It's right skewed, mean > median

paste0("Mean price: ",mean(house_data$price))
paste0("Median price: ",median(house_data$price))

#correlation
res <- cor(house_data)
corrplot(res,method="number")

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(res, method="color", col=col(200),  
         type="upper", order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
        sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE , number.cex = 0.4, tl.cex = 0.5
)

house_data_reduced <- house_data[,!(names(house_data) %in% c("",""))]

formula_based_on_rpart_importance = price ~ feat02 + feat05 + view + grade + sqft_above + sqft_living +sqft_living15 + bathrooms + lat + long + yr_built + floors + condition+waterfront+bedrooms;
#deal with outliers in key variables
important_features <- c("feat02","feat05","view","grade","sqft_above","sqft_living15","sqft_living","bathrooms","bedrooms","lat","long","yr_built","floors","condition")

for (i in important_features) {
  quantiles <- quantile(house_data[[i]], c(0.01, 0.99))
  house_data <- house_data %>%
    mutate({{i}} := ifelse(house_data[[i]] < quantiles[1], quantiles[1], 
                           ifelse(house_data[[i]] > quantiles[2], quantiles[2], house_data[[i]])))
}

table(house_data_init$yr_built)
table(house_data$yr_built)
quantiles <- quantile(house_data$grade, c(0.01, 0.99))

# Cap outliers
house_data <- house_data %>%
  mutate(column_name = ifelse(column_name < quantiles[1], quantiles[1], 
                              ifelse(column_name > quantiles[2], quantiles[2], column_name)))

#delete outliers in Y var
quartiles <- quantile(house_data$price, probs=c(.25, .99), na.rm = FALSE)
IQR <- IQR(house_data$price)
Lower <- quartiles[1] - 1.5*IQR
Upper <- quartiles[2] + 1.5*IQR 
data_no_outlier <- subset(house_data, house_data$price < Upper)


####plot density
b <- ggplot(data_no_outlier, aes(x = price))
b + geom_density() +
  geom_vline(aes(xintercept = mean(price)), 
             linetype = "dashed", size = 0.6) + scale_x_continuous()








#MAP
# Install and load required libraries
#install.packages("leaflet")
library(leaflet)

# Sort the data by price in descending order
sorted_data <- house_data[order(-house_data$price), ]

top
# Select the top 30 houses with the highest prices
top_100_high_price <- head(sorted_data, 200)

# Select the top 30 houses with the lowest prices
top_100_low_price <- tail(sorted_data, 200)
top_99_quantile_high_price <- subset(sorted_data, price>=Upper)
# Combine the selected dataframes
selected_data <- rbind(top_100_high_price, top_100_low_price)

# Define colors for the map
selected_data$markerColor <- ifelse(selected_data$price %in% top_99_quantile_high_price$price, "gold", ifelse(selected_data$price %in% top_100_high_price$price,"blue","green"))

map <- leaflet(selected_data) %>%
  addTiles() %>%
  addCircleMarkers(
    lng = ~long,
    lat = ~lat,
    label = ~price,
    radius = 5,
    color = ~markerColor,
    fillOpacity = 0.8
  ) %>%
  addLegend("bottomright", colors = c("gold","blue" ,"green"), labels = c("TOP 99th quantile $$$","Top 200 $$$", "The Cheapest 200 $"), opacity = 1)

map





#################### data partition

set.seed(123456789)
training_obs <- createDataPartition(house_data$price, 
                                    p = 0.7, 
                                    list = FALSE)
House.train <- house_data[training_obs,]
House.test  <- house_data[-training_obs,]

#Balanced data
House.train.balanced <- SmoteRegress(formula_based_on_rpart_importance, House.train,rel = "auto", thr.rel = 0.01, C.perc = "extreme",
                                     k = 5, repl = FALSE, dist = "Euclidean", p = 2)
c <- ggplot(House.train.balanced, aes(x = price))
c + geom_density() +
  geom_vline(aes(xintercept = mean(price)), 
             linetype = "dashed", size = 0.6) + scale_x_continuous()



training_obs2 <- createDataPartition(data_no_outlier$price, 
                                    p = 0.7, 
                                    list = FALSE)
House.train2 <- data_no_outlier[training_obs2,]
House.test2  <- data_no_outlier[-training_obs2,]



#general glm model

House.glmFull <- glm(formula_based_on_rpart_importance, data=House.train)
summary(House.glmFull)


House.glmFull.pred.train <- predict(House.glmFull, House.train)
(lm.MSE.train <- sum((House.glmFull.pred.train - House.train$price) ^ 2) / nrow(House.train))
(lm.MAE.train <- sum(abs(House.glmFull.pred.train - House.train$price)) / nrow(House.train))

House.glmFull.pred.test <- predict(House.glmFull, House.test)
(lm.MSE.test <- sum((House.glmFull.pred.test - House.test$price) ^ 2) / nrow(House.test))
(lm.MAE.test <- sum(abs(House.glmFull.pred.test - House.test$price)) / nrow(House.test))

formula_based_on_glm_importance <- price ~ bathrooms + bedrooms + condition + date + feat02 + feat05 + grade + lat + long + sqft_above + sqft_basement + sqft_living15  + sqft_lot15 + view + waterfront + yr_built + yr_renovated;


### tree model

House.treeFull <- tree(formula_based_on_rpart_importance, House.train)

House.treeFull
summary(House.treeFull)

#MSE for train
House.treeFull.TrainPred  <- predict(House.treeFull,  newdata = House.train)
mean((House.treeFull.TrainPred- as.numeric(House.train$price)) ^ 2)
#MAE for train
mean(abs(House.treeFull.TrainPred- as.numeric(House.train$price)))

#MSE for test
House.treeFull.TestPred  <- predict(House.treeFull,  newdata = House.test)
paste("MSE for TEST = ",mean((House.treeFull.TestPred- House.test$price) ^ 2))
#MAE for test
paste("MAE for TEST =",mean(abs(House.treeFull.TestPred- as.numeric(House.test$price))))

sd(House.train$price) ^ 2 * (length(House.train$price) - 1)
(House.train$price - mean(House.train$price))  ^ 2 %>% sum

plot(House.treeFull)
text(House.treeFull, pretty = 0)


######## tree model reduced
House.treeReduced <- tree(formula_based_on_glm_importance, data=House.train)
summary(House.treeReduced)

House.treeReduced.TrainPred  <- predict(House.treeReduced,  newdata = House.train)
mean((House.treeReduced.TrainPred- House.train$price) ^ 2)
mean(abs(House.treeReduced.TrainPred- House.train$price))

House.treeReduced2 <- tree(formula_based_on_rpart_importance, data=House.train)
summary(House.treeReduced2)

House.treeReduced2.TrainPred  <- predict(House.treeReduced2,  newdata = House.train)
mean((House.treeReduced2.TrainPred- House.train$price) ^ 2)
mean(abs(House.treeReduced2.TrainPred- House.train$price))
#on data without outliers
House.treeReducedWithoutOutliers2 <- tree(formula_based_on_rpart_importance, data=House.train2)
summary(House.treeReducedWithoutOutliers2)

House.treeReducedWithoutOutliers2.TrainPred  <- predict(House.treeReducedWithoutOutliers2,  newdata = House.train2)
paste("MSE for Train Tree reduced = ",mean((House.treeReducedWithoutOutliers2.TrainPred- House.train2$price) ^ 2))
paste("MAE for Train Tree reduced = ",mean(abs(House.treeReducedWithoutOutliers2.TrainPred- House.train2$price)))

# 5 pruned tree ================================================================
House.cv <- cv.tree(House.treeReduced2, K = 15)
plot(House.cv$size, House.cv$dev, type = 'b')
House.treeReduced2.pruned <- prune.tree(House.treeReduced2, best = 12)
House.treeReduced2.pruned.pred <- predict(House.treeReduced2.pruned,
                              newdata = House.test)
mean((House.treeReduced2.pruned.pred   - House.test$price) ^ 2)
mean(abs(House.treeReduced2.pruned.pred   - House.test$price))
##############################

###GLM REDUCED  
House.glmReducedWithoutOutliers <- glm(formula_based_on_rpart_importance, data=House.train2)
summary(House.glmReducedWithoutOutliers)
#MSE for train
sum(House.glmReducedWithoutOutliers$residuals^2) / House.glmReducedWithoutOutliers$df.residual
#MAE for train
sum(abs(House.glmReducedWithoutOutliers$residuals)) / House.glmReducedWithoutOutliers$df.residual
#################### RPART TREE


House.treeRPARTReduced <- rpart(formula_based_on_rpart_importance,
                     data = House.train,
                     method = "anova")
view(House.treeRPARTReduced$cptable)
summary(House.treeRPARTReduced)

#var importance
df <- data.frame(imp = House.treeRPARTReduced$variable.importance)
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))
ggplot2::ggplot(df2) +
  geom_col(aes(x = variable, y = imp),
           col = "black", show.legend = F) +
  coord_flip() +
  scale_fill_grey() +
  theme_bw()





hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)
models <- list()
for (i in 1:nrow(hyper_grid)) {
  
  cat(i, "/", nrow(hyper_grid), "\n", sep = "")
  
  # setting the values
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  # settin the seed
  set.seed(123123 + i)
  
  # training of the model and saving results to the list
  models[[i]] <- rpart(
    formula = formula_based_on_rpart_importance,
    data    = House.train.balanced,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}

get_cp <- function(x) {
  min <- which.min(x$cptable[, "xerror"])
  cp  <- x$cptable[min, "CP"] 
  return(cp)
}

get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
  return(xerror)
}
hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

houses.tree.optimal <- rpart(
  formula = formula_based_on_rpart_importance,
  data    = House.train.balanced,
  method  = "anova",
  control = list(minsplit = 5, maxdepth = 15, cp = 0.01)
)

pred <- predict(houses.tree.optimal, newdata = House.train.balanced)
#(houses.tree.optimal.RMSE <- RMSE(pred = pred, obs = House.train$price))
mean((pred   - House.train.balanced$price) ^ 2)
mean(abs(pred   - House.train.balanced$price))

pred_test <- predict(houses.tree.optimal, newdata = House.test)
#(houses.tree.optimal.RMSE <- RMSE(pred = pred, obs = House.train$price))
mean((pred_test   - House.test$price) ^ 2)
mean(abs(pred_test   - House.test$price))

#RF
library(randomForest)

kingCounty.rf <- randomForest(formula_based_on_rpart_importance ,
                          data   = House.train.balanced,
                          mtry   = 5, ntree=300,nodesize=3,
                          importance = TRUE)

kingCounty.rf.pred <- predict(kingCounty.rf,
                          newdata = House.train.balanced)
mean((kingCounty.rf.pred   - House.train.balanced$price) ^ 2)
mean(abs(kingCounty.rf.pred   - House.train.balanced$price))
kingCounty.rf.predt <- predict(kingCounty.rf,
                          newdata = House.test)
mean((kingCounty.rf.predt   - House.test$price) ^ 2)
mean(abs(kingCounty.rf.predt   - House.test$price))


#########Gradient Boosting
House.gbm <- gbm(formula_based_on_rpart_importance, 
                  data = House.train, 
                  distribution = "gaussian",
                  n.trees = 600, #800
                  interaction.depth = 10, shrinkage=0.01)
House.gbm.pred <- predict(House.gbm,
                           newdata = House.train,
                           n.trees = 600,
                          interaction.depth = 10, shrinkage=0.01) #shrinkage is a key to disable overfitting
paste("MSE for TRAIN GBM = ",mean((House.gbm.pred - House.train$price) ^ 2))
paste("MAE for TRAIN GBM = ",mean(abs(House.gbm.pred - House.train$price)))

House.gbm.test_pred <- predict(House.gbm,
                                newdata = House.test,
                                n.trees = 600,
                               interaction.depth = 10, shrinkage=0.01)
paste("MSE for TEST GBM = ",mean((House.gbm.test_pred - House.test$price) ^ 2))
paste("MAE for TEST GBM = ",mean(abs(House.gbm.test_pred - House.test$price)))

#Balanced boosting
House_balanced.gbm <- gbm(formula_based_on_rpart_importance, 
                 data = House.train.balanced, 
                 distribution = "gaussian",
                 n.trees = 800,
                 interaction.depth = 10, shrinkage = 0.01)
House_balanced.gbm.pred <- predict(House_balanced.gbm,
                          newdata = House.train.balanced,
                          n.trees = 800,
                          interaction.depth = 10, shrinkage = 0.01)
paste("MSE for TRAIN GBM = ",mean((House_balanced.gbm.pred - House.train.balanced$price) ^ 2))
paste("MAE for TRAIN GBM = ",mean(abs(House_balanced.gbm.pred - House.train.balanced$price)))

House_balanced.gbm.test_pred <- predict(House_balanced.gbm,
                               newdata = House.test,
                               n.trees = 800,
                               interaction.depth = 10, shrinkage = 0.01)
paste("MSE for TEST GBM = ",mean((House_balanced.gbm.test_pred - House.test$price) ^ 2))
paste("MAE for TEST GBM = ",mean(abs(House_balanced.gbm.test_pred - House.test$price)))




################Smaller model
House.gbm_reduced1 <- gbm(price ~ grade + sqft_above + sqft_living15 + bathrooms + yr_built + floors + lat + long + condition, 
                 data = House.train, 
                 distribution = "gaussian",
                 n.trees = 500,
                 interaction.depth = 4)
House.gbm_reduced1.pred <- predict(House.gbm_reduced1,
                          newdata = House.train,
                          n.trees = 500)
paste("MSE for TRAIN GBM = ",mean((House.gbm_reduced1.pred - House.train$price) ^ 2))
paste("MAE for TRAIN GBM = ",mean(abs(House.gbm_reduced1.pred - House.train$price)))

House.gbm_reduced1.test_pred <- predict(House.gbm_reduced1,
                               newdata = House.test,
                               n.trees = 500)
paste("MSE for TEST GBM = ",mean((House.gbm_reduced1.test_pred - House.test$price) ^ 2))
paste("MAE for TEST GBM = ",mean(abs(House.gbm_reduced1.test_pred - House.test$price)))


###############################
House2.gbm <- gbm(formula_based_on_rpart_importance, 
                  data = House.train2, 
                  distribution = "gaussian",
                  n.trees = 800,
                  interaction.depth = 10, shrinkage = 0.01)
summary(House2.gbm)
House2.gbm.pred <- predict(House2.gbm,
                           newdata = House.train2,
                           n.trees = 800,
                           interaction.depth = 10, shrinkage = 0.01)
paste("MSE for TRAIN GBM = ",mean((House2.gbm.pred - House.train2$price) ^ 2))
paste("MAE for TRAIN GBM = ",mean(abs(House2.gbm.pred - House.train2$price)))

House2.gbm.test_pred <- predict(House2.gbm,
                           newdata = House.test2,
                           n.trees = 800,
                           interaction.depth = 10, shrinkage = 0.01)
paste("MSE for TEST GBM = ",mean((House2.gbm.test_pred - House.test2$price) ^ 2))
paste("MAE for TEST GBM = ",mean(abs(House2.gbm.test_pred - House.test2$price)))

House.gbm.test_pred2 <- predict(House2.gbm,
                                newdata = House.test,
                                n.trees = 800,
                                interaction.depth = 10)
paste("MSE for TEST GBM = ",mean((House.gbm.test_pred2 - House.test$price) ^ 2))
paste("MAE for TEST GBM = ",mean(abs(House.gbm.test_pred2 - House.test$price)))



# 3 gradient boosting tuned ====================================================
if (1) {
  grid <- expand.grid(interaction.depth = c(3, 7),
                      n.trees = c(100, 500, 1000),
                      shrinkage = c(0.01, 0.1), 
                      n.minobsinnode = c(5, 7, 10, 15))
  
  House.gbm.tuned  <- train(formula_based_on_rpart_importance,
                             data = House.train,
                             distribution = "gaussian",
                             method = "gbm",
                             tuneGrid = grid,
                             verbose = FALSE)  
  
}
saveRDS(House.gbm.tuned, file= "House.gbm.tuned.rds")
House.gbm.tuned <- readRDS("House.gbm.tuned.rds")

# optimal parameters:
House.gbm.tuned
# n.trees = 1000, interaction.depth = 7,
# shrinkage = 0.01 and n.minobsinnode = 10.

# retraining the model with optimal parameters
House.gbm3 <- gbm(formula_based_on_rpart_importance ,
                   data = House.train,
                   distribution = "gaussian",
                   n.trees = 1000,
                   interaction.depth = 7,
                   shrinkage = 0.01,
                   n.minobsinnode = 10,
                   verbose = F)
House.gbm3.train.pred <- predict(House.gbm3,
                           newdata = House.train,
                           n.trees = 1000)
mean((House.gbm3.train.pred - House.train$price) ^ 2)
mean(abs(House.gbm3.train.pred - House.train$price))

House.gbm3.pred <- predict(House.gbm3,
                            newdata = House.test,
                            n.trees = 1000)
mean((House.gbm3.pred - House.test$price) ^ 2)
mean(abs(House.gbm3.pred - House.test$price))



######### Neural Network

#Benchmark LM
lm.fit <- lm(formula_based_on_rpart_importance , data = House.train)
summary(lm.fit)
formula_based_on_rpart_importance
lm.pred.train <- predict(lm.fit, House.train)
(lm.MSE.train <- sum((lm.pred.train - House.train$price) ^ 2) / nrow(House.train))
(lm.MAE.train <- sum(abs(lm.pred.train - House.train$price)) / nrow(House.train))

lm.pred <- predict(lm.fit, House.test)
(lm.MSE <- sum((lm.pred - House.test$price) ^ 2) / nrow(House.test))
(lm.MAE <- sum(abs(lm.pred - House.test$price)) / nrow(House.test))

train.maxs <- apply(House.train, 2, max)
train.mins <- apply(House.train, 2, min)


train.maxs2 <- apply(House.train2, 2, max)
train.mins2 <- apply(House.train2, 2, min)

train.maxs3 <- apply(House.train.balanced, 2, max)
train.mins3 <- apply(House.train.balanced, 2, min)

#Standarization
House.train %>% glimpse()
House.train.scaled <- 
  as.data.frame(scale(House.train, 
                      center = train.mins, 
                      scale  = train.maxs - train.mins))
House.test.scaled <- 
  as.data.frame(scale(House.test, 
                      center = train.mins, 
                      scale  = train.maxs - train.mins))



House.train2.scaled <- 
  as.data.frame(scale(House.train2, 
                      center = train.mins2, 
                      scale  = train.maxs2 - train.mins2))
House.test2.scaled <- 
  as.data.frame(scale(House.test2, 
                      center = train.mins2, 
                      scale  = train.maxs2 - train.mins2))


House.train3.scaled <- 
  as.data.frame(scale(House.train.balanced, 
                      center = train.mins3, 
                      scale  = train.maxs3 - train.mins3))




nn1 <- neuralnet(formula_based_on_rpart_importance, 
                data   = House.train.scaled,
                # number of neurons in the hidden layer
                hidden = c(1), 
                # T for regression, F for classification
                linear.output = T, 
                threshold = 0.01,
                learningrate.limit = NULL,
                learningrate.factor = list(minus = 0.5, plus = 1.2),
                algorithm = "rprop+")

nn1_2 <- neuralnet(formula_based_on_rpart_importance, 
                data   = House.train.scaled,
                hidden = c(1), 
                linear.output = T)
#Plot of neural network
plot(nn1_2, rep = "best")

nn1_2.train.pred <- compute(nn1_2,subset(House.train.scaled, select=-price))
#scalling back
nn1_2.train.pred.unscaled <- 
  nn1_2.train.pred$net.result * 
  (train.maxs["price"] - train.mins["price"]) + train.mins["price"]
nn1_2.train.MSE <- sum((House.train$price - nn1_2.train.pred.unscaled)^2)/nrow(House.train)
nn1_2.train.MAE <- sum(abs(House.train$price - nn1_2.train.pred.unscaled))/nrow(House.train)
cat(paste0("Train MSE.lm = ", round(lm.MSE.train, 2), ", " ,  
           "Train MSE.nn = ", round(nn1_2.train.MSE, 2)))
cat(paste0("Train MAE.lm = ", round(lm.MAE.train, 2), ", " ,  
           "Train MAE.nn = ", round(nn1_2.train.MAE, 2)))


nn1_2.test.pred <- compute(nn1_2, subset(House.test.scaled, select=-price))
nn1_2.test.pred.unscaled <- 
  nn1_2.test.pred$net.result * 
  (train.maxs["price"] - train.mins["price"]) + train.mins["price"]
nn1_2.test.MSE <- sum((House.test$price - nn1_2.test.pred.unscaled)^2)/nrow(House.test)
nn1_2.test.MAE <- sum(abs(House.test$price - nn1_2.test.pred.unscaled))/nrow(House.test)
cat(paste0("Test MSE.lm = ", round(lm.MSE, 2), ", " ,  
           "Test MSE.nn = ", round(nn1_2.test.MSE, 2)))
cat(paste0("Test MAE.lm = ", round(lm.MAE, 2), ", " ,  
           "Test MAE.nn = ", round(nn1_2.test.MAE, 2)))

#plot
plot(House.test$price,
     nn1_2.test.pred.unscaled, 
     col = 'red', main = 'Real vs predicted NN', pch = 18, cex = 0.7)
points(House.test$price, 
       lm.pred, 
       col = 'blue', pch = 18, cex = 0.7)
abline(0, 1, lwd = 2)
legend('topleft', legend = c('NN', 'LM'), 
       pch = 18, col = c('red', 'blue'))


##bigger NN

nn2 <- neuralnet(formula_based_on_rpart_importance, 
                 data   = House.train.scaled,
                 hidden = c(10), 
                 linear.output = T, threshold=0.2,stepmax=1e4)

saveRDS(nn2, file= "nn2.rds")
nn2 <- readRDS("nn2.rds")

plot(nn2, rep = "best")


nn2.train.pred <- compute(nn2,subset(House.train.scaled, select=-price))
#scalling back
nn2.train.pred.unscaled <- 
  nn2.train.pred$net.result * 
  (train.maxs["price"] - train.mins["price"]) + train.mins["price"]
nn2.train.MSE <- sum((House.train$price - nn2.train.pred.unscaled)^2)/nrow(House.train)
nn2.train.MAE <- sum(abs(House.train$price - nn2.train.pred.unscaled))/nrow(House.train)
cat(paste0("Train MSE.lm = ", round(lm.MSE.train, 2), ", " ,  
           "Train MSE.nn = ", round(nn2.train.MSE, 2)))
cat(paste0("Train MAE.lm = ", round(lm.MAE.train, 2), ", " ,  
           "Train MAE.nn = ", round(nn2.train.MAE, 2)))


nn2.test.pred <- compute(nn2, subset(House.test.scaled, select=-price))
nn2.test.pred.unscaled <- 
  nn2.test.pred$net.result * 
  (train.maxs["price"] - train.mins["price"]) + train.mins["price"]
nn2.test.MSE <- sum((House.test$price - nn2.test.pred.unscaled)^2)/nrow(House.test)
nn2.test.MAE <- sum(abs(House.test$price - nn2.test.pred.unscaled))/nrow(House.test)
cat(paste0("Test MSE.lm = ", round(lm.MSE, 2), ", " ,  
           "Test MSE.nn = ", round(nn2.test.MSE, 2)))
cat(paste0("Test MAE.lm = ", round(lm.MAE, 2), ", " ,  
           "Test MAE.nn = ", round(nn2.test.MAE, 2)))

#plot
plot(House.test$price,
     nn2.test.pred.unscaled, 
     col = 'red', main = 'Real vs predicted NN', pch = 18, cex = 0.7)
points(House.test$price, 
       lm.pred, 
       col = 'blue', pch = 18, cex = 0.7)
abline(0, 1, lwd = 2)
legend('topleft', legend = c('NN', 'LM'), 
       pch = 18, col = c('red', 'blue'))



#Next NN

nn3 <- neuralnet(formula_based_on_rpart_importance, 
                 data   = House.train.scaled,
                 hidden = c(20), 
                 linear.output = T, threshold=0.2)

saveRDS(nn3, file= "nn3.rds")
nn3 <- readRDS("nn3.rds")

plot(nn3, rep = "best")


nn3.train.pred <- compute(nn3,subset(House.train.scaled, select=-price))
#scalling back
nn3.train.pred.unscaled <- 
  nn3.train.pred$net.result * 
  (train.maxs["price"] - train.mins["price"]) + train.mins["price"]
nn3.train.MSE <- sum((House.train$price - nn3.train.pred.unscaled)^2)/nrow(House.train)
nn3.train.MAE <- sum(abs(House.train$price - nn3.train.pred.unscaled))/nrow(House.train)
cat(paste0("Train MSE.lm = ", round(lm.MSE.train, 2), ", " ,  
           "Train MSE.nn = ", round(nn3.train.MSE, 2)))
cat(paste0("Train MAE.lm = ", round(lm.MAE.train, 2), ", " ,  
           "Train MAE.nn = ", round(nn3.train.MAE, 2)))


nn3.test.pred <- compute(nn3, subset(House.test.scaled, select=-price))
nn3.test.pred.unscaled <- 
  nn3.test.pred$net.result * 
  (train.maxs["price"] - train.mins["price"]) + train.mins["price"]
nn3.test.MSE <- sum((House.test$price - nn3.test.pred.unscaled)^2)/nrow(House.test)
nn3.test.MAE <- sum(abs(House.test$price - nn3.test.pred.unscaled))/nrow(House.test)
cat(paste0("Test MSE.lm = ", round(lm.MSE, 2), ", " ,  
           "Test MSE.nn = ", round(nn3.test.MSE, 2)))
cat(paste0("Test MAE.lm = ", round(lm.MAE, 2), ", " ,  
           "Test MAE.nn = ", round(nn3.test.MAE, 2)))

#plot
plot(House.test$price,
     nn3.test.pred.unscaled, 
     col = 'red', main = 'Real vs predicted NN', pch = 18, cex = 0.7)
points(House.test$price, 
       lm.pred, 
       col = 'blue', pch = 18, cex = 0.7)
abline(0, 1, lwd = 2)
legend('topleft', legend = c('NN', 'LM'), 
       pch = 18, col = c('red', 'blue'))







##NN for dataset all outliers done

#Next NN

nn3_2 <- neuralnet(formula_based_on_rpart_importance, 
                 data   = House.train2.scaled,
                 hidden = c(20), 
                 linear.output = T, threshold=0.2)

saveRDS(nn3_2, file= "nn3_2.rds")
nn3_2 <- readRDS("nn3_2.rds")




nn3_2.train.pred <- compute(nn3_2,subset(House.train2.scaled, select=-price))
#scalling back
nn3_2.train.pred.unscaled <- 
  nn3_2.train.pred$net.result * 
  (train.maxs2["price"] - train.mins2["price"]) + train.mins2["price"]
nn3_2.train.MSE <- sum((House.train2$price - nn3_2.train.pred.unscaled)^2)/nrow(House.train2)
nn3_2.train.MAE <- sum(abs(House.train2$price - nn3_2.train.pred.unscaled))/nrow(House.train2)
cat(paste0("Train MSE.lm = ", round(lm.MSE.train, 2), ", " ,  
           "Train MSE.nn = ", round(nn3_2.train.MSE, 2)))
cat(paste0("Train MAE.lm = ", round(lm.MAE.train, 2), ", " ,  
           "Train MAE.nn = ", round(nn3_2.train.MAE, 2)))


nn3_2.test.pred <- compute(nn3_2, subset(House.test, select=-price))
nn3_2.test.pred.unscaled <- 
  nn3_2.test.pred$net.result * 
  (train.maxs2["price"] - train.mins2["price"]) + train.mins2["price"]
nn3_2.test.MSE <- sum((House.test$price - nn3_2.test.pred.unscaled)^2)/nrow(House.test)
nn3_2.test.MAE <- sum(abs(House.test$price - nn3_2.test.pred.unscaled))/nrow(House.test)
cat(paste0("Test MSE.lm = ", round(lm.MSE, 2), ", " ,  
           "Test MSE.nn = ", round(nn3_2.test.MSE, 2)))
cat(paste0("Test MAE.lm = ", round(lm.MAE, 2), ", " ,  
           "Test MAE.nn = ", round(nn3_2.test.MAE, 2)))



##NN for dataset balanced

#Next NN

nn3_3 <- neuralnet(formula_based_on_rpart_importance, 
                   data   = House.train3.scaled,
                   hidden = c(20), 
                   linear.output = T, threshold=0.2)

saveRDS(nn3_3, file= "nn3_3.rds")
nn3_3 <- readRDS("nn3_3.rds")




nn3_3.train.pred <- compute(nn3_3,subset(House.train3.scaled, select=-price))
#scalling back
nn3_3.train.pred.unscaled <- 
  nn3_3.train.pred$net.result * 
  (train.maxs3["price"] - train.mins3["price"]) + train.mins3["price"]
nn3_3.train.MSE <- sum((House.train.balanced$price - nn3_3.train.pred.unscaled)^2)/nrow(House.train.balanced)
nn3_3.train.MAE <- sum(abs(House.train.balanced$price - nn3_3.train.pred.unscaled))/nrow(House.train.balanced)
cat(paste0("Train MSE.lm = ", round(lm.MSE.train, 2), ", " ,  
           "Train MSE.nn = ", round(nn3_3.train.MSE, 2)))
cat(paste0("Train MAE.lm = ", round(lm.MAE.train, 2), ", " ,  
           "Train MAE.nn = ", round(nn3_3.train.MAE, 2)))


nn3_3.test.pred <- compute(nn3_3, subset(House.test, select=-price))
nn3_3.test.pred.unscaled <- 
  nn3_3.test.pred$net.result * 
  (train.maxs2["price"] - train.mins2["price"]) + train.mins2["price"]
nn3_3.test.MSE <- sum((House.test$price - nn3_3.test.pred.unscaled)^2)/nrow(House.test)
nn3_3.test.MAE <- sum(abs(House.test$price - nn3_3.test.pred.unscaled))/nrow(House.test)
cat(paste0("Test MSE.lm = ", round(lm.MSE, 2), ", " ,  
           "Test MSE.nn = ", round(nn3_3.test.MSE, 2)))
cat(paste0("Test MAE.lm = ", round(lm.MAE, 2), ", " ,  
           "Test MAE.nn = ", round(nn3_3.test.MAE, 2)))




##Kable Table output

Model <- c("GLM","TREE","RF","GBM","NN")
Train_MSE <- c(lm.MSE.train,mean((pred   - House.train.balanced$price) ^ 2),mean((kingCounty.rf.pred   - House.train.balanced$price) ^ 2),mean((House2.gbm.pred - House.train2$price) ^ 2),nn3.train.MSE)
Train_MAE <- c(lm.MAE.train,mean(abs(pred   - House.train.balanced$price)) ,mean(abs(kingCounty.rf.pred   - House.train.balanced$price)),mean(abs(House2.gbm.pred - House.train2$price)),nn3.train.MAE)
Test_MSE <- c(lm.MSE.test,mean((pred_test   - House.test$price) ^ 2),mean((kingCounty.rf.predt   - House.test$price) ^ 2),mean((House2.gbm.test_pred - House.test2$price) ^ 2),nn3.test.MSE)
Test_MAE <- c(lm.MAE.test,mean(abs(pred_test   - House.test$price)),mean(abs(kingCounty.rf.predt   - House.test$price)),mean(abs(House2.gbm.test_pred - House.test2$price)),nn3.test.MAE)

final_results <- data.frame(Model, Train_MSE, Train_MAE, Test_MSE, Test_MAE)


final_results$Train_MSE <- paste0(round((final_results$Train_MSE/1000000),0)," M")
final_results$Test_MSE <- paste0(round((final_results$Test_MSE/1000000),0)," M")
final_results$Train_MAE <- paste0(round((final_results$Train_MAE/1000),0)," K")
final_results$Test_MAE <- paste0(round((final_results$Test_MAE/1000),0)," K")
final_results
####### NN with keras

#Reccurent

#Now, we will train and evaluate a densely connected model. We use MAE for the loss function.

library(reticulate)
version <- "3.9.12"
install_python(version)
virtualenv_create("my-environment", version = version)
use_virtualenv("my-environment")

# There is also support for a ":latest" suffix to select the latest patch release
#install_python("3.9:latest") # install latest patch available at python.org

# select the latest 3.9.* patch installed locally
#virtualenv_create("my-environment", version = "3.9:latest")

py_install("tensorflow")
py_install("keras")


library(tensorflow)
install_tensorflow()
library(keras)

sequence_length <- 10
n_samples <- nrow(House.train.scaled)
X <- array(0, dim = c(n_samples - sequence_length + 1, sequence_length, 8))
y <- House.train.scaled$price[sequence_length:n_samples]

for (i in 1:(n_samples - sequence_length + 1)) {
  X[i,,] <- cbind(House.train.scaled$grade[i:(i+sequence_length-1)], House.train.scaled$sqft_above[i:(i+sequence_length-1)],House.train.scaled$sqft_living15[i:(i+sequence_length-1)],House.train.scaled$bathrooms[i:(i+sequence_length-1)],House.train.scaled$lat[i:(i+sequence_length-1)],House.train.scaled$long[i:(i+sequence_length-1)],House.train.scaled$yr_built[i:(i+sequence_length-1)],House.train.scaled$floors[i:(i+sequence_length-1)])
}

rnn_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(sequence_length, 8)) %>%
  layer_dense(units = 1)  # 1 output node for regression


model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam()
)

model %>% fit(
  X, y,
  epochs = 50,
  batch_size = 32,
  verbose = 1
)

evaluation <- model %>% evaluate(X_test, y_test)
print(evaluation)
