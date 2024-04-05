
# Loading Required Libraries
library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(tidyverse)
library(caret)

# Loading the Dataset
credit_card_data <- read_csv("C:/Users/Lana/Documents/CSU/MIS581/creditcard_2023.csv")

# Viewing the First Few Rows of the Dataset
head(credit_card_data)

#Checking for missing values
colSums(is.na(credit_card_data))

# Describing the Dataset
str(credit_card_data)

# Calculate Summary Metrics for 'Time' and 'Amount' only
summary_metrics_time_amount <- credit_card_data %>%
  summarise(across(c(Time, Amount), list(
    Mean = ~mean(., na.rm = TRUE),
    SD = ~sd(., na.rm = TRUE),
    Min = ~min(., na.rm = TRUE),
    Max = ~max(., na.rm = TRUE)
  ))) %>%
  pivot_longer(cols = everything(), names_to = "Variable_Metric", values_to = "Value") %>%
  separate(Variable_Metric, into = c("Variable", "Metric"), sep = "_") %>%
  pivot_wider(names_from = Variable, values_from = Value)

# Display the summary table
print(summary_metrics_time_amount)

# Normalize 'Time' and 'Amount'
preProcessRange <- preProcess(credit_card_data[, c("Time", "Amount")], method = c("center", "scale"))
data_norm <- predict(preProcessRange, credit_card_data[, c("Time", "Amount")])

# Removing the original 'Time' and 'Amount' columns and Appending the normalized
credit_card_data <- bind_cols(credit_card_data[, -which(names(credit_card_data) %in% c("Time", "Amount"))], data_norm)

# Class Distribution
class_distribution <- table(credit_card_data$Class)
print(class_distribution)

# Plot Class Distribution
ggplot(credit_card_data, aes(x = factor(Class), fill = factor(Class))) +
  geom_bar() +
  labs(x = "Class", y = "Count", fill = "Class", title = "Distribution of Classes") +
  scale_fill_manual(values = c("0" = "grey", "1" = "orange"), labels = c("Genuine", "Fraudulent")) +
  scale_y_continuous(labels = scales::comma) +  # Use comma formatting for y-axis labels
  theme_minimal()

# Transaction Amount Analysis
ggplot(credit_card_data, aes(x = Amount, fill = factor(Class))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("0" = "blue", "1" = "orange"), labels = c("Genuine", "Fraudulent")) +
  labs(x = "Transaction Amount", y = "Density", fill = "Class", title = "Transaction Amount by Class") +
  theme_minimal()

# Adjusted Transaction Amount Analysis with Log Scale
ggplot(credit_card_data, aes(x = Amount, fill = factor(Class))) +
  geom_density(alpha = 0.5) +
  scale_x_log10(labels = scales::dollar) +  # Apply log scale transformation to x-axis and format labels as dollars
  scale_fill_manual(values = c("0" = "blue", "1" = "orange"), labels = c("Genuine", "Fraudulent")) +
  labs(x = "Transaction Amount (Log Scale)", y = "Density", fill = "Class", title = "Transaction Amount by Class (Log Scale)") +
  theme_minimal()

# Transaction Time Analysis
ggplot(credit_card_data, aes(x = Time, fill = factor(Class))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 100) +
  scale_fill_manual(values = c("0" = "blue", "1" = "orange"), labels = c("Genuine", "Fraudulent")) +
  labs(x = "Time (in seconds)", y = "Count", fill = "Class", title = "Transactions Over Time") +
  theme_minimal()

# Faceted Transaction Time Analysis
ggplot(credit_card_data, aes(x = Time, fill = factor(Class))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 100) +
  facet_wrap(~ Class, scales = "free_y", ncol = 1) +  # Facet by Class, allowing different y scales
  scale_fill_manual(values = c("0" = "blue", "1" = "orange"), labels = c("Genuine", "Fraudulent")) +
  labs(x = "Time (in seconds)", y = "Count", fill = "Class", title = "Transaction Time Distribution by Class") +
  theme_minimal()

# Load necessary library
library(ROSE)

# Split the data into training and test sets (e.g., 70% training, 30% test)
set.seed(123)  # Setting seed for reproducibility
indexes <- createDataPartition(credit_card_data$Class, p = 0.7, list = FALSE)
training_set <- credit_card_data[indexes, ]
test_set <- credit_card_data[-indexes, ]

# Calculate the number of majority class samples
num_majority <- sum(training_set$Class == 0)

# Calculate the number of minority class samples
num_minority <- sum(training_set$Class == 1)

# We want to oversample the minority class to have the same number of samples as the majority class
desired_n <- num_majority + num_minority  # Total desired number of samples after oversampling

# Use ovun.sample to oversample the minority class to match the number of majority class
training_set_balanced <- ovun.sample(Class ~ ., data = training_set, method = "over",
                                     N = 2*num_majority, seed = 123)$data

# Check the balance of the new training set
table(training_set_balanced$Class)

# Load necessary libraries
library(e1071)
library(nnet)
library(rpart)
library(randomForest)
library(PRROC)
library(MLmetrics)
library(pROC)

# Ensure 'Class' is a factor and rename its levels
training_set_balanced$Class <- factor(training_set_balanced$Class, levels = c(0, 1), labels = c("Negative", "Positive"))
test_set$Class <- factor(test_set$Class, levels = c(0, 1), labels = c("Negative", "Positive"))

# Set up control for training using cross-validation
train_control <- trainControl(method = "cv", number = 10, summaryFunction = prSummary, classProbs = TRUE)


## LOGISTIC REGRESSION
set.seed(123)
lr_model <- train(Class ~ ., data = training_set_balanced, method = "glm",
                  family = "binomial", trControl = train_control, metric = "AUCPR")
print(lr_model)

# Predictions on test data
lr_predictions <- predict(lr_model, newdata = test_set, type = "prob")

# Convert actual class labels to binary format for ROC Curve
test_set_actual_binary <- ifelse(test_set$Class == "Positive", 1, 0)

# Extract the predicted probabilities for the "Positive" class for both curves
lr_predictions_positive <- lr_predictions[, "Positive"]

# Generate ROC curve data for manual plotting
roc_data <- roc(test_set_actual_binary, lr_predictions_positive, plot = FALSE)

# Calculate the True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity)
tpr <- roc_data$sensitivities
fpr <- 1 - roc_data$specificities


# Generate the Precision-Recall curve object
pr_result <- pr.curve(scores.class0 = lr_predictions_positive, weights.class0 = test_set_actual_binary, curve = TRUE)

# Extract precision and recall values
precision_values <- pr_result$curve[,2]
recall_values <- pr_result$curve[,1]


# Set graphical parameters to plot side by side
par(mfrow = c(1, 2))

# Base R plot for ROC curve with custom axes
plot(fpr, tpr, type = "l", col = "blue", lwd = 2,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC Curve for Logistic Regression")
abline(a = 0, b = 1, lty = 2, col = "gray")

# Plot the Precision-Recall Curve
plot(recall_values, precision_values, type = "l", col = "blue", lwd = 2,
     xlab = "Recall", ylab = "Precision",
     main = "Precision-Recall Curve for Logistic Regression")

# Reset graphical parameters to default
par(mfrow = c(1, 1))

# Assuming auc_lr and auprc_lr hold the AUROC and AUPRC values for the logistic regression model
# (calculate these values as shown in previous steps if not already done)
auc_lr <- auc(roc_data)  # Make sure this is computed if not already
auprc_lr <- pr_result$auc.integral  # Make sure this is computed if not already

# Create a data frame to hold the model evaluation metrics
model_eval_metrics <- data.frame(
  Model = "Logistic Regression",
  AUROC = auc_lr,
  AUPRC = auprc_lr
)

# Print the table
print(model_eval_metrics)

## DECISION TREE 
library(rpart)

# Fit Decision Tree model
set.seed(123)  # For reproducibility
dt_model <- rpart(Class ~ ., data = training_set_balanced, method = "class")

# Print the summary of the Decision Tree model
print(summary(dt_model))

# Predictions on test data
dt_predictions <- predict(dt_model, newdata = test_set, type = "prob")

# Generate ROC curve data
dt_roc_curve <- roc(response = test_set_actual_binary, predictor = dt_predictions[, "Positive"], quiet = TRUE)

# Extract the True Positive Rate and False Positive Rate
dt_tpr <- dt_roc_curve$sensitivities
dt_fpr <- 1 - dt_roc_curve$specificities

# Generate Precision-Recall curve data
dt_pr_curve <- pr.curve(scores.class0 = dt_predictions[, "Positive"], weights.class0 = test_set_actual_binary, curve = TRUE)

# Extract precision and recall values
dt_precision_values <- dt_pr_curve$curve[,2]
dt_recall_values <- dt_pr_curve$curve[,1]

# Set graphical parameters to plot side by side
par(mfrow = c(1, 2))

# Base R plot for ROC curve with custom axes
plot(dt_fpr, dt_tpr, type = "l", col = "blue", lwd = 2,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC Curve for Decision Tree",
     ylim = c(0, 1))  # Set y-axis limits to 0 and 1
abline(a = 0, b = 1, lty = 2, col = "gray")  # Diagonal line for reference

# Add AUROC value inside the plot
text(x = 0.5, y = 0.5, labels = paste("AUROC:", round(0.9213749, 2)), pos = 3, cex = 0.8, col = "black")

# Base R plot for Precision-Recall curve with custom axes
plot(dt_recall_values, dt_precision_values, type = "l", col = "blue", lwd = 2,
     xlab = "Recall", ylab = "Precision",
     main = "Precision-Recall Curve for Decision Tree",
     ylim = c(0, 1))  # Set y-axis limits to 0 and 1

# Add AUPRC value inside the plot
text(x = 0.5, y = 0.5, labels = paste("AUPRC:", round(0.04221578, 2)), pos = 3, cex = 0.8, col = "black")

# Reset graphical parameters to default
par(mfrow = c(1, 1))


# AUROC
dt_roc_result <- roc(response = test_set_actual_binary, predictor = dt_predictions[, "Positive"], quiet = TRUE)
dt_auroc <- auc(dt_roc_result)

# AUPRC using PRROC package
dt_pr_result <- pr.curve(scores.class0 = dt_predictions[, "Positive"], weights.class0 = test_set_actual_binary, curve = TRUE)
dt_auprc <- dt_pr_result$auc.integral

# Add the Decision Tree model metrics to the table
dt_model_metrics <- data.frame(
  Model = "Decision Tree",
  AUROC = dt_auroc,
  AUPRC = dt_auprc
)
model_eval_metrics <- rbind(model_eval_metrics, dt_model_metrics)

# Print the updated table
print(model_eval_metrics)


## RANDOM FOREST
library(randomForest)

# Train Random Forest model
set.seed(123)  # For reproducibility
rf_model <- randomForest(Class ~ ., data = training_set_balanced, ntree = 100)  # Using 100 trees

# Print the model summary
print(rf_model)

# Predictions on test data
rf_predictions <- predict(rf_model, newdata = test_set, type = "prob")

# AUROC
rf_roc_result <- roc(response = test_set_actual_binary, predictor = rf_predictions[, "Positive"], quiet = TRUE)
rf_auroc <- auc(rf_roc_result)

# AUPRC
rf_pr_result <- pr.curve(scores.class0 = rf_predictions[, "Positive"], weights.class0 = test_set_actual_binary, curve = TRUE)
rf_auprc <- rf_pr_result$auc.integral

# Extract the True Positive Rate and False Positive Rate from ROC result
rf_tpr <- rf_roc_result$sensitivities
rf_fpr <- 1 - rf_roc_result$specificities

# Extract precision and recall values from PR result
rf_precision_values <- rf_pr_result$curve[, 2]
rf_recall_values <- rf_pr_result$curve[, 1]

# Set graphical parameters to plot side by side
par(mfrow = c(1, 2))

# ROC Curve for Random Forest with custom axes
plot(rf_fpr, rf_tpr, type = "l", col = "blue", lwd = 2,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC Curve for Random Forest",
     ylim = c(0, 1))  # Set y-axis limits to 0 and 1
abline(a = 0, b = 1, lty = 2, col = "gray")  # Diagonal line for reference

# Add AUROC value inside the plot
text(x = 0.5, y = 0.5, labels = paste("AUROC:", round(0.9494944, 2)), pos = 3, cex = 0.8, col = "black")

# Precision-Recall Curve for Random Forest with custom axes
plot(rf_recall_values, rf_precision_values, type = "l", col = "blue", lwd = 2,
     xlab = "Recall", ylab = "Precision",
     main = "Precision-Recall Curve for Random Forest",
     ylim = c(0, 1))  # Set y-axis limits to 0 and 1

# Add AUPRC value inside the plot
text(x = 0.5, y = 0.5, labels = paste("AUPRC:", round(0.84399240, 2)), pos = 3, cex = 0.8, col = "black")

# Reset graphical parameters to default
par(mfrow = c(1, 1))


# Add the Random Forest model metrics to the table
rf_model_metrics <- data.frame(
  Model = "Random Forest",
  AUROC = rf_auroc,
  AUPRC = rf_auprc
)
model_eval_metrics <- rbind(model_eval_metrics, rf_model_metrics)

# Print the updated table
print(model_eval_metrics)


## NEURAL NETWORK

library(nnet)

set.seed(123)  # For reproducibility
nn_model <- nnet(Class ~ ., data = training_set_balanced, size = 10, linout = FALSE, maxit = 200)

# Print the model summary
print(nn_model)

# Ensure 'Class' column is excluded if it's the target variable and not an input feature
nn_predictions_probs <- predict(nn_model, newdata=test_set[, !names(test_set) %in% "Class"], type="raw")
nn_predictions <- ifelse(nn_predictions_probs > 0.5, 1, 0)

# AUROC
nn_roc_result <- roc(response = as.numeric(test_set$Class), predictor = nn_predictions_probs)
nn_auroc <- auc(nn_roc_result)

# AUPRC
nn_pr_result <- pr.curve(scores.class0 = nn_predictions_probs, weights.class0 = as.numeric(test_set$Class) == 1, curve = TRUE)
nn_auprc <- nn_pr_result$auc.integral

# Extract metrics for ROC Curve
nn_tpr <- nn_roc_result$sensitivities
nn_fpr <- 1 - nn_roc_result$specificities

# Extract metrics for Precision-Recall Curve
nn_precision_values <- nn_pr_result$curve[, 2]
nn_recall_values <- nn_pr_result$curve[, 1]

# Set graphical parameters to plot side by side
par(mfrow = c(1, 2))

# ROC Curve for Neural Network with custom axes
plot(nn_fpr, nn_tpr, type = "l", col = "blue", lwd = 2,
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "ROC Curve for Neural Network",
     ylim = c(0, 1))  # Set y-axis limits to 0 and 1
abline(a = 0, b = 1, lty = 2, col = "gray")  # Diagonal reference line

# Add AUROC value inside the plot
text(x = 0.5, y = 0.5, labels = paste("AUROC:", round(0.9692691, 2)), pos = 3, cex = 0.8, col = "black")

# Precision-Recall Curve for Neural Network with custom axes
plot(nn_recall_values, nn_precision_values, type = "l", col = "blue", lwd = 2,
     xlab = "Recall", ylab = "Precision",
     main = "Precision-Recall Curve for Neural Network",
     ylim = c(0, 1))  # Set y-axis limits to 0 and 1

# Add AUPRC value inside the plot
text(x = 0.5, y = 0.5, labels = paste("AUPRC:", round(0.98955824, 2)), pos = 3, cex = 0.8, col = "black")

# Reset graphical parameters to default
par(mfrow = c(1, 1))


# Add the Nueral Network model metrics to the table
model_eval_metrics <- rbind(model_eval_metrics, data.frame(Model = "Neural Network", AUROC = nn_auroc, AUPRC = nn_auprc))
print(model_eval_metrics)


# Set graphical parameters to plot side by side
par(mfrow = c(1, 2))

# ROC Curves Overlay
plot(fpr, tpr, type = "l", col = "blue", lwd = 2,
     xlab = "False Positive Rate",
     ylab = "True Positive Rate",
     main = "ROC Curves")
lines(dt_fpr, dt_tpr, type = "l", col = "orange", lwd = 2)
lines(rf_fpr, rf_tpr, type = "l", col = "purple", lwd = 2)
lines(nn_fpr, nn_tpr, type = "l", col = "brown", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "Random Forest", "Neural Network"),
       col = c("blue", "orange", "purple", "brown"), lwd = 2)

# Precision-Recall Curves Overlay
plot(recall_values, precision_values, type = "l", col = "blue", lwd = 2, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Recall", ylab = "Precision",
     main = "Precision-Recall Curves")
lines(dt_recall_values, dt_precision_values, type = "l", col = "orange", lwd = 2)
lines(rf_recall_values, rf_precision_values, type = "l", col = "purple", lwd = 2)
lines(nn_recall_values, nn_precision_values, type = "l", col = "brown", lwd = 2)

# Reset graphical parameters to default
par(mfrow = c(1, 1))

# Create a dataframe with the provided data
data <- data.frame(
  Model = c("Log Reg", "Dec Tree", "Ran For", "Neu Net"),
  AUROC = c(0.9808144, 0.9213749, 0.9494944, 0.9692691),
  AUPRC = c(0.74885022, 0.04221578, 0.84399240, 0.98955824)
)

# Set graphical parameters to plot side by side
par(mfrow = c(1, 2))

# Create bar plot for AUROC
barplot(data$AUROC, names.arg = data$Model, col = c("blue", "orange", "purple", "brown"), main = "AUROC", ylab = "AUROC Value", las = 2, ylim = c(0, 1))

# Create bar plot for AUPRC
barplot(data$AUPRC, names.arg = data$Model, col = c("blue", "orange", "purple", "brown"), main = "AUPRC", ylab = "AUPRC Value", las = 2, ylim = c(0, 1))

# Reset graphical parameters to default
par(mfrow = c(1, 1))



