# Importing libraries
library(caret) # Package for machine learning algorithms

# Importing the Iris data set
irs <- datasets::iris
irs <- data.frame(irs)

# missing values
# sum(is.na(irs))

# apply column transformation
col_names_transformation <- function(x){
   new_names = names(x)
   new_names = gsub("\\.", "", new_names)
   
   names(x) = new_names
   return(x)
}

irs <- col_names_transformation(irs)

# To achieve reproducible model; set the random seed number
set.seed(189)

# Performs stratified random split of the data set
TrainingIndex <- createDataPartition(irs$Species, p=0.8, list = FALSE)
TrainingSet <- irs[TrainingIndex,] # Training Set
TestingSet <- irs[-TrainingIndex,] # Test Set


###############################
# k-Nearest Neighbors (kNN) Model

# Build kNN Training model
knn_Model <- train(Species ~ ., data = TrainingSet,
                   method = "knn",
                   na.action = na.omit,
                   preProcess=c("scale","center"),
                   trControl= trainControl(method="none"),
                   tuneGrid = data.frame(k=5) # You can experiment with different k values
)

# Apply kNN model for prediction
knn_Model.training <- predict(knn_Model, TrainingSet)
knn_Model.testing <- predict(knn_Model, TestingSet)
