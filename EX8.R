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


sequence_training_set <- read.csv('Sequences_train.csv', sep = ',', header = FALSE, stringsAsFactors = FALSE)
sequence_testing_set <- read.csv('Sequences_test_unlabeled.csv', sep = ',', header = FALSE, stringsAsFactors = FALSE)

sequence_training_data <- sequence_training_set[, 1]
sequence_training_labels <- sequence_training_set[, 2]

sequence_training_labels <- replace(sequence_training_labels, sequence_training_labels == -1, 0)
# sequence_training_labels <- to_categorical(sequence_training_labels)


acid_length <- nchar(sequence_training_data[1])
alphabet <- get_alphabet()
encoding_matrix <- calculate_encoded_matrix(sequence_training_data)

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 300, activation = "relu", input_shape = c(300)) %>% 
  # layer_dense(units = 400, activation = "relu") %>%
  # layer_dense(units = 100, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  # loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  
  # loss = "mean_squared_error",
  # optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

history <- model %>% fit(
  encoding_matrix, sequence_training_labels, 
  epochs = 40, batch_size = 128, 
  validation_split = 0.2
)

print(history)


# ff <- calculate_encoded_matrix(sequence_training_set[1, 1])
# print(model %>% predict_classes(ff))


###################
# sigmoid
# categorical_crossentropy
# rmsprop
# 89.5
###################
# sigmoid
# mean squared
# rmsprop
# 88.25
##################