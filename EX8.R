# Name: Omar Amr
# Matrikel-Nr: k11776960

library(keras)
# install_keras()
# install.packages('curl')


set.seed(123)

directory <- '~/Desktop/ML-Assign-6-EX-8'
setwd(directory)


get_alphabet <- function()
{
  result <- vector()
  for(acid in sequence_training_data)
  {
    temp <- unique(unlist(strsplit(acid, "")))
    result <- unique(append(result, temp))
  }
  result <- sort(result)
  return(result)
}



encode_sequence <- function(input_sequence)
{
  data_sample_split <- unlist(strsplit(input_sequence, ""))
  data_sample_vect <- vector()
  data_sample_encoding <- replicate(length(alphabet), "0")
  for (char in data_sample_split)
  {
    temp_encoding <- replicate(length(alphabet), "0")
    temp_encoding[match(char, alphabet)] <- 1
    temp_encoding <- unlist(strsplit(temp_encoding, ""))
    data_sample_vect <- append(data_sample_vect, temp_encoding)
  }
  return(as.integer(data_sample_vect))
}



calculate_encoded_matrix <- function(dataset)
{
  matrix_size <- length(dataset)
  result <- matrix(nrow=matrix_size, ncol=acid_length*length(alphabet),byrow =TRUE)
  for (data_sample_counter in 1:matrix_size)
  {
    result[data_sample_counter, ] <- encode_sequence(dataset[data_sample_counter])
  }
  print("Data encoding done")
  return(result)
}

predict_test_sample_1 <- function(test_data)
{
  prediction_results <- vector()
  for (x in test_data) 
  {
    temp <- calculate_encoded_matrix(x)
    temp <- model %>% predict_classes(temp)
    prediction_results <- c(prediction_results, temp)
  }
  prediction_results <- replace(prediction_results, prediction_results == 0, -1)
  write(prediction_results, "NN_predictions_1_hidden_layer.txt", sep="\n")
  print("Test sample prediction done!")
}

predict_test_sample_2 <- function(test_data)
{
  prediction_results <- vector()
  for (x in test_data) 
  {
    temp <- calculate_encoded_matrix(x)
    temp <- model2 %>% predict_classes(temp)
    prediction_results <- c(prediction_results, temp)
  }
  prediction_results <- replace(prediction_results, prediction_results == 0, -1)
  write(prediction_results, "NN_predictions_2_hidden_layers.txt", sep="\n")
  print("Test sample prediction done!")
}


search_for_best_params_1_layer <- function()
{
  best_accuracy <- 0
  first_hidden_layer_vector <- c(50, 75, 100, 200, 250)
  learning_rate_vector <- c(0.0001, 0.0005, 0.001, 0.005, 0.01)
  dropout_rate_vector <- c(0.2, 0.3, 0.4, 0.5, 0.6)
  
  best_first_hidden_layer <- 0
  best_learning_rate <- 0
  best_dropout_rate <- 0
  
  for (current_number_of_hidden_nodes in first_hidden_layer_vector)
  {
    for (current_learning_rate in learning_rate_vector)
    {
      for (current_dropout_rate in dropout_rate_vector)
      {
        print(paste0("Best accuracy so far: ", best_accuracy))
        print(paste0("Current number of hidden nodes: ", current_number_of_hidden_nodes))
        print(paste0("Current learning rate: ", current_learning_rate))
        print(paste0("Current dropout rate: ", current_dropout_rate))
        
        temp_start_time <- Sys.time()
        
        temp_model <- keras_model_sequential()
        
        temp_model %>%
          layer_dense(units = 300, activation = "relu", input_shape = c(300)) %>%
          layer_dropout(rate = current_dropout_rate) %>%
          layer_dense(units = current_number_of_hidden_nodes, activation = "relu") %>%
          layer_dropout(rate = current_dropout_rate) %>%
          layer_dense(units = 1, activation = "sigmoid")
        
        temp_model %>% compile(
          loss = "binary_crossentropy",
          optimizer = optimizer_rmsprop(lr = current_learning_rate),
          metrics = c("accuracy")
        )
        
        epoch_number <- 100
        history <- temp_model %>% fit(
          encoding_matrix, sequence_training_labels,
          epochs = epoch_number, batch_size = 40,
          validation_split = 0.2
        )
        
        if(history$metrics$val_acc[epoch_number] > best_accuracy)
        {
          best_first_hidden_layer <- current_number_of_hidden_nodes
          best_learning_rate <- current_learning_rate
          best_dropout_rate <- current_dropout_rate
          best_accuracy <- history$metrics$val_acc[epoch_number]
        }
        
        temp_end_time <- Sys.time()
        print(paste0("Current Model Time: ", temp_end_time - temp_start_time))
      }
    }
  }
  
  print(paste0("Overall Best Accuracy: ", best_accuracy))
  print(paste0("Best Number of Nodes: ", best_first_hidden_layer))
  print(paste0("Best Dropout: ", best_dropout_rate))
  print(paste0("Best Learning Rate: ", best_learning_rate))
  
  train_1_hidden_layer_model(hidden_nodes_number=best_first_hidden_layer, 
                             learning_rate=best_learning_rate, dropout_rate=best_dropout_rate)
}

train_1_hidden_layer_model <- function(hidden_nodes_number, learning_rate, dropout_rate)
{
  model <<- keras_model_sequential()
  
  model %>%
    layer_dense(units = 300, activation = "relu", input_shape = c(300)) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = hidden_nodes_number, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = learning_rate),
    metrics = c("accuracy")
  )
  
  epoch_number <- 100
  history <- model %>% fit(
    encoding_matrix, sequence_training_labels,
    epochs = epoch_number, batch_size = 40,
    validation_split = 0.2
  )
  
  print(history)
}


search_for_best_params_2_layers <- function()
{
  best_accuracy <- 0
  first_hidden_layer_vector <- c(50, 75, 100, 200, 250)
  second_hidden_layer_vector <- c(50, 75, 100, 200, 250)
  learning_rate_vector <- c(0.0001, 0.0005, 0.001, 0.005, 0.01)
  dropout_rate_vector <- c(0.2, 0.3, 0.4, 0.5, 0.6)
  
  best_first_hidden_layer <- 0
  best_second_hidden_layer <- 0
  best_learning_rate <- 0
  best_dropout_rate <- 0
  
  for (current_number_of_hidden_nodes_layer1 in first_hidden_layer_vector) 
  {
    for (current_number_of_hidden_nodes_layer2 in second_hidden_layer_vector)
    {
      for (current_learning_rate in learning_rate_vector)
      {
        for (current_dropout_rate in dropout_rate_vector)
        {
          print(paste0("Best accuracy so far: ", best_accuracy))
          print(paste0("Current number of hidden nodes 1: ", current_number_of_hidden_nodes_layer1))
          print(paste0("Current number of hidden nodes 2: ", current_number_of_hidden_nodes_layer2))
          print(paste0("Current learning rate: ", current_learning_rate))
          print(paste0("Current dropout rate: ", current_dropout_rate))
          
          temp_start_time <- Sys.time()
          
          temp_model <- keras_model_sequential()
          
          temp_model %>%
            layer_dense(units = 300, activation = "relu", input_shape = c(300)) %>%
            layer_dropout(rate = current_dropout_rate) %>%
            layer_dense(units = current_number_of_hidden_nodes_layer1, activation = "relu") %>%
            layer_dropout(rate = current_dropout_rate) %>%
            layer_dense(units = current_number_of_hidden_nodes_layer2, activation = "relu") %>%
            layer_dropout(rate = current_dropout_rate) %>%
            layer_dense(units = 1, activation = "sigmoid")
          
          temp_model %>% compile(
            loss = "binary_crossentropy",
            optimizer = optimizer_rmsprop(lr = current_learning_rate),
            metrics = c("accuracy")
          )
          
          epoch_number <- 100
          history <- temp_model %>% fit(
            encoding_matrix, sequence_training_labels,
            epochs = epoch_number, batch_size = 40,
            validation_split = 0.2
          )
          
          if(history$metrics$val_acc[epoch_number] > best_accuracy)
          {
            best_first_hidden_layer <- current_number_of_hidden_nodes_layer1
            best_second_hidden_layer <- current_number_of_hidden_nodes_layer2
            best_learning_rate <- current_learning_rate
            best_dropout_rate <- current_dropout_rate
            best_accuracy <- history$metrics$val_acc[epoch_number]
          }
          
          temp_end_time <- Sys.time()
          print(paste0("Current Model Time: ", temp_end_time - temp_start_time))
        }
      }
    }  
  }
  
  
  
  print(paste0("Overall Best Accuracy: ", best_accuracy))
  print(paste0("Best Number of Nodes 1: ", best_first_hidden_layer))
  print(paste0("Best Number of Nodes 2: ", best_second_hidden_layer))
  print(paste0("Best Dropout: ", best_dropout_rate))
  print(paste0("Best Learning Rate: ", best_learning_rate))
  
  train_2_hidden_layer_model(hidden_nodes_number_1=best_first_hidden_layer, hidden_nodes_number_2 = best_second_hidden_layer,
                             learning_rate=best_learning_rate, dropout_rate=best_dropout_rate)
}

train_2_hidden_layer_model <- function(hidden_nodes_number_1, hidden_nodes_number_2,
                           learning_rate, dropout_rate)
{
  model2 <<- keras_model_sequential()
  
  model2 %>%
    layer_dense(units = 300, activation = "relu", input_shape = c(300)) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = hidden_nodes_number_1, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = hidden_nodes_number_2, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model2 %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = learning_rate),
    metrics = c("accuracy")
  )
  
  epoch_number <- 100
  history <- model2 %>% fit(
    encoding_matrix, sequence_training_labels,
    epochs = epoch_number, batch_size = 40,
    validation_split = 0.2
  )
  
  print(history)
}

sequence_training_set <- read.csv('Sequences_train.csv', sep = ',', header = FALSE, stringsAsFactors = FALSE)
sequence_testing_set <- read.csv('Sequences_test_unlabeled.csv', sep = ',', header = FALSE, stringsAsFactors = FALSE)

sequence_training_data <- sequence_training_set[, 1]
sequence_training_labels <- sequence_training_set[, 2]
sequence_training_labels <- replace(sequence_training_labels, sequence_training_labels == -1, 0)

acid_length <- nchar(sequence_training_data[1])
alphabet <- get_alphabet()
encoding_matrix <- calculate_encoded_matrix(sequence_training_data)

start_time <- Sys.time()


###################################################################################################
#  Search for best parameters for NN with 1 hidden layer and train a model with these parameters  #
###################################################################################################

search_for_best_params_1_layer()




#####################################################
#  Train 1-hidden-layer Model With Best Parameters  #
#####################################################

#train_1_hidden_layer_model(hidden_nodes_number=100, learning_rate=0.001, dropout_rate=0.4) # 90%
# predict_test_sample_1(sequence_testing_set)

# TODO:
# search_for_best_params_2_layers()
# train_2_hidden_layer_model(hidden_nodes_number_1 = 200, hidden_nodes_number_2 = 100, learning_rate = 0.001, dropout_rate = 0.3) # 91%
# predict_test_sample_2(sequence_testing_set)


##########################
#  Predict Test Dataset  #
##########################




end_time <- Sys.time()
print(paste0("Total Time: ", end_time - start_time))
