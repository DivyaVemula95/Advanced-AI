# Remove all objects from the cluster to ensure it's clean
h2o.removeAll()
# Lists all objects in the H2O cluster
h2o.ls()
# Shutdown the H2O cluster
h2o.shutdown(prompt = FALSE)

# Downloading dependencies
install.packages(c('skimr', 'recipes', 'stringr', 'tidyverse', 'h2o'))


# Importing required packages
library(skimr)
library(recipes)
library(stringr)
library(tidyverse)
library(h2o)

# Loading car pricing prediction Dataset
carPrice_prediction <- read_csv("Assignment3-Vemula-CarPricingPrediction.csv")

# Removed unimportant Date columns and cleaned the data
carPrice_prediction <- carPrice_prediction |> select(-dateCrawled, -dateCreated, -lastSeen)
carPrice_prediction <- carPrice_prediction |> drop_na(price)
glimpse(carPrice_prediction)

# Transforming the price value
carPrice_prediction$price <- log1p(carPrice_prediction$price)

# partitioning the dataset into x and y tibbles
train_x_tbl <- carPrice_prediction |> select(-price)
train_y_tbl <- carPrice_prediction |> select(price)

# Remove the original data to save memory
rm(carPrice_prediction)

# skimr
train_x_tbl_skim = skim(train_x_tbl)
print(train_x_tbl_skim)

# Factorizing Character Data
# Get the names of all character columns in the dataset
string_2_factor_names <- train_x_tbl_skim$character$skim_variable

# Create a recipe to preprocess the data
rec_obj <- recipe(~ ., data = train_x_tbl) |>
  step_string2factor(all_of(string_2_factor_names)) |>
  step_impute_median(all_numeric()) |>  # Handle missing values in numeric columns
  step_impute_mode(all_nominal()) |>   # Handle missing values in factor columns
  prep()


# Bake using recipe
train_x_processed_tbl <- bake(rec_obj, train_x_tbl)
glimpse(train_x_processed_tbl)
View(train_x_processed_tbl)

# Writing processed data set into a csv file
final_data <- cbind(train_x_processed_tbl, train_y_tbl)
write.csv(final_data, "Assignment3-Vemula-CarPricingPrediction.csv", row.names = FALSE, append = FALSE)

# Building ML model using H2O
h2o.init(nthreads = -1) #-1 to use all cores

# push data into h2o; NOTE: THIS MAY TAKE A WHILE!
data_h2o <- as.h2o(final_data)

# split the data into training, validation and test sets
splits <- h2o.splitFrame(data = data_h2o, seed = 1234,
                         ratios = c(0.7, 0.15)) # 70/15/15 split
train_h2o <- splits[[1]] # from training data
valid_h2o <- splits[[2]] # from training data
test_h2o <- splits[[3]] # from training data

y <- "price"
x <- setdiff(names(train_h2o), y)

# training the model
## Random Hyper-Parameter Search
hyper_params <- list(
  ntrees = c(5, 10, 20),
  max_depth = c(10, 20, 30),
  min_rows = c(5, 10, 15),
  sample_rate = c(0.7, 0.8)
)



## Random forest Model
rf_model <- h2o.grid(
  model_id = "Assignment3-Vemula-CarPricingPrediction",
  algorithm = "randomForest",
  grid_id = "rf_grid_random",
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  hyper_params = hyper_params
)

# Summary Of Random forest model
summary(rf_model, show_stack_traces = TRUE)

ordered_grid <- h2o.getGrid("rf_grid_random", sort_by="rmse",  decreasing=F)
dl_grid_random_summary_table <- ordered_grid@summary_table
View(dl_grid_random_summary_table)

# Best Model
dl_grid_random_best_model <- h2o.getModel(dl_grid_random_summary_table$model_ids[1])
summary(dl_grid_random_best_model)

# Saving the h2o model
h2o.saveModel(object = dl_grid_random_best_model, path = getwd(), force = TRUE, filename = "Assignment3-Vemula-CarPricingPrediction.h2o")

h2o.shutdown(prompt = F)

