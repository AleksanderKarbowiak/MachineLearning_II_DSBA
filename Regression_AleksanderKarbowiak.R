library(tidyverse)
library(tree)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(here)
library(corrplot)

setwd(here())
getwd()
house_data <- read.csv(file="r1.csv",header=TRUE,sep=",")
house_data_init <- house_data

str(house_data)
colSums(is.na(house_data)) %>% 
  sort()


#distribution of price var
a <- ggplot(house_data, aes(x = price))
a + geom_density() +
  geom_vline(aes(xintercept = mean(price)), 
             linetype = "dashed", size = 0.6) + scale_x_continuous()


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
         diag=FALSE 
)

house_data_reduced <- house_data[,!(names(house_data) %in% c("",""))]


#deal with outliers in key variables
important_features <- c("grade","sqft_above","sqft_living15","bathrooms","lat","long","yr_built")

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
quartiles <- quantile(house_data$price, probs=c(.25, .9), na.rm = FALSE)
IQR <- IQR(house_data$price)
Lower <- quartiles[1] - 1.5*IQR
Upper <- quartiles[2] + 1.5*IQR 
data_no_outlier <- subset(house_data, house_data$price < Upper)


####plot density
b <- ggplot(data_no_outlier, aes(x = price))
b + geom_density() +
  geom_vline(aes(xintercept = mean(price)), 
             linetype = "dashed", size = 0.6) + scale_x_continuous()


#################### data partition

set.seed(123456789)
training_obs <- createDataPartition(house_data$price, 
                                    p = 0.7, 
                                    list = FALSE)
House.train <- house_data[training_obs,]
House.test  <- house_data[-training_obs,]

training_obs2 <- createDataPartition(data_no_outlier$price, 
                                    p = 0.7, 
                                    list = FALSE)
House.train2 <- data_no_outlier[training_obs2,]
House.test2  <- data_no_outlier[-training_obs2,]



#general glm model

House.glmFull <- glm(price ~ ., data=House.train)
summary(House.glmFull)
formula_based_on_glm_importance <- price ~ bathrooms + bedrooms + condition + date + feat02 + feat05 + grade + lat + long + sqft_above + sqft_basement + sqft_living15  + sqft_lot15 + view + waterfront + yr_built + yr_renovated;


### tree model

House.treeFull <- tree(price ~ ., House.train)

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

House.treeReduced2 <- tree(formula_based_on_rpart_importance, data=House.train)
summary(House.treeReduced2)

House.treeReduced2.TrainPred  <- predict(House.treeReduced2,  newdata = House.train)
mean((House.treeReduced2.TrainPred- House.train$price) ^ 2)

#on data without outliers
House.treeReducedWithoutOutliers2 <- tree(formula_based_on_rpart_importance, data=House.train2)
summary(House.treeReducedWithoutOutliers2)

House.treeReducedWithoutOutliers2.TrainPred  <- predict(House.treeReducedWithoutOutliers2,  newdata = House.train2)
paste("MSE for Train Tree reduced = ",mean((House.treeReducedWithoutOutliers2.TrainPred- House.train2$price) ^ 2))
paste("MAE for Train Tree reduced = ",mean(abs(House.treeReducedWithoutOutliers2.TrainPred- House.train2$price)))

# 5 pruned tree ================================================================
House.cv <- cv.tree(House.treeReducedWithoutOutliers2, K = 15)
plot(House.cv$size, House.cv$dev, type = 'b')
House.treeReducedWithoutOutliers2.pruned <- prune.tree(House.treeReducedWithoutOutliers2, best = 11)
House.treeReducedWithoutOutliers2.pruned.pred <- predict(House.treeReducedWithoutOutliers2.pruned,
                              newdata = House.test)
mean((House.treeReducedWithoutOutliers2.pruned.pred   - House.test$price) ^ 2)
##############################

###GLM REDUCED  
House.glmReducedWithoutOutliers <- glm(formula_based_on_rpart_importance, data=House.train2)
summary(House.glmReducedWithoutOutliers)
#MSE for train
sum(House.glmReducedWithoutOutliers$residuals^2) / House.glmReducedWithoutOutliers$df.residual
#MAE for train
sum(abs(House.glmReducedWithoutOutliers$residuals)) / House.glmReducedWithoutOutliers$df.residual
#################### RPART TREE

set.seed(45564556)
House.treeRPARTReduced <- rpart(formula_based_on_glm_importance,
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

formula_based_on_rpart_importance = price ~ grade + sqft_above + sqft_living15 + bathrooms + lat + long + yr_built;



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
    formula = price ~ .,
    data    = House.train,
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






#########Gradient Boosting
library(gbm)
House.gbm <- gbm(formula_based_on_rpart_importance, 
                  data = House.train, 
                  distribution = "gaussian",
                  n.trees = 500,
                  interaction.depth = 4)
House.gbm.pred <- predict(House.gbm,
                           newdata = House.train,
                           n.trees = 500)
paste("MSE for TRAIN GBM = ",mean((House.gbm.pred - House.train$price) ^ 2))
paste("MAE for TRAIN GBM = ",mean(abs(House.gbm.pred - House.train$price)))

House.gbm.test_pred <- predict(House.gbm,
                                newdata = House.test,
                                n.trees = 500)
paste("MSE for TEST GBM = ",mean((House.gbm.test_pred - House.test$price) ^ 2))
paste("MAE for TEST GBM = ",mean(abs(House.gbm.test_pred - House.test$price)))


House2.gbm <- gbm(formula_based_on_rpart_importance, 
                  data = House.train2, 
                  distribution = "gaussian",
                  n.trees = 500,
                  interaction.depth = 4)
summary(House2.gbm)
House2.gbm.pred <- predict(House2.gbm,
                           newdata = House.train2,
                           n.trees = 500)
paste("MSE for TRAIN GBM = ",mean((House2.gbm.pred - House.train2$price) ^ 2))
paste("MAE for TRAIN GBM = ",mean(abs(House2.gbm.pred - House.train2$price)))

House2.gbm.test_pred <- predict(House2.gbm,
                           newdata = House.test2,
                           n.trees = 500)
paste("MSE for TEST GBM = ",mean((House2.gbm.test_pred - House.test2$price) ^ 2))
paste("MAE for TEST GBM = ",mean(abs(House2.gbm.test_pred - House.test2$price)))

House.gbm.test_pred2 <- predict(House2.gbm,
                                newdata = House.test,
                                n.trees = 500)
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
House.gbm.tuned <- readRDS(here("output", "House.gbm.tuned.rds"))

# optimal parameters:
House.gbm.tuned
# n.trees = 1000, interaction.depth = 7,
# shrinkage = 0.01 and n.minobsinnode = 10.

# retraining the model with optimal parameters
set.seed(seed)
House.gbm3 <- gbm(formula_based_on_rpart_importance ,
                   data = House.train,
                   distribution = "gaussian",
                   n.trees = 1000,
                   interaction.depth = 7,
                   shrinkage = 0.01,
                   n.minobsinnode = 10,
                   verbose = F)
House.gbm3.pred <- predict(House.gbm3,
                            newdata = House.test,
                            n.trees = 1000)
mean((House.gbm3.pred - House.test$price) ^ 2)
mean(abs(House.gbm3.pred - House.test$price))
