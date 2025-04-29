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

# Model performance (Displays confusion matrix and statistics)
model.training.confmtx <- confusionMatrix(knn_Model.training, TrainingSet$Species)


# Calculate additional metrics: Accuracy, Precision, Recall, F1 Score
calculate_metrics <- function(conf_matrix) {
   accuracy <- conf_matrix$overall["Accuracy"]
   precision <- mean(conf_matrix$byClass[, "Precision"])
   recall <- mean(conf_matrix$byClass[, "Recall"])
   f1_score <- 2 * (precision * recall) / (precision + recall)
   
   metrics <- data.frame(
      Metric = c("Accuracy", "Precision", "Recall", "F1 Score"),
      Value = c(accuracy, precision, recall, f1_score)
   )
   return(metrics)
}

# Checking metrics
calculate_metrics(model.training.confmtx)

# Visualize the confusion matrix
visualize_confusion_matrix <- function(conf_matrix, title) {
   cm <- as.data.frame(conf_matrix$table)
   colnames(cm) <- c("Actual", "Prediction", "Freq")
   
   ggplot(cm, aes(x = Prediction, y = Actual)) +
      geom_tile(aes(fill = Freq), color = "white") +
      scale_fill_gradient(low = "white", high = "blue") +
      geom_text(aes(label = Freq), vjust = 1) +
      ggtitle(title) +
      theme_minimal()
}

visualize_confusion_matrix(model.training.confmtx, "Training Set Confusion Matrix.")