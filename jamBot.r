# install.packages('keras')
# install.packages('tuneR')
# install_keras()

# Install and load necessary libraries
if (!requireNamespace("keras", quietly = TRUE)) install.packages("keras")
if (!requireNamespace("tuneR", quietly = TRUE)) install.packages("tuneR")
if (!requireNamespace("tensorflow", quietly = TRUE)) install.packages("tensorflow")
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")

library(keras)
library(tuneR)
library(seewave)
library(signal)
library(dplyr)
library(tensorflow)
library(reticulate)

# Function to load and preprocess recorded music
load_recorded_data <- function(audio_folder) {
  audio_files <- list.files(audio_folder, pattern = "\\.wav$", full.names = TRUE)
  
  # Print the current working directory and list of audio files found
  print(paste("Current Working Directory:", getwd()))
  print("Audio files found:")
  print(audio_files)
  
  if (length(audio_files) == 0) {
    stop("No WAV files found in the specified directory.")
  }
  
  sound_data_list <- lapply(audio_files, function(audio_file) {
    sound <- readWave(audio_file)
    sound_data <- sound@left
    return(sound_data)
  })
  
  sound_data <- unlist(sound_data_list)
  sample_rate <- readWave(audio_files[1])@samp.rate
  return(list(data = sound_data, sample_rate = sample_rate))
}

# Example pitch shift function (you may need to adjust this or find an appropriate package)
pitchshift <- function(sound_data, semitones) {
  # Placeholder function: you need to replace this with actual pitch shifting implementation
  return(sound_data)
}

# Example time stretch function (you may need to adjust this or find an appropriate package)
stretch <- function(sound_data, factor) {
  # Placeholder function: you need to replace this with actual time stretching implementation
  return(sound_data)
}

# Data augmentation function
augment_data <- function(sound_data) {
  augmented_data <- list()
  
  for (data in sound_data) {
    # Pitch shift
    augmented_data <- c(augmented_data, list(data))
    augmented_data <- c(augmented_data, list(pitchshift(data, semitones = 2)))
    augmented_data <- c(augmented_data, list(pitchshift(data, semitones = -2)))
    
    # Time stretch
    augmented_data <- c(augmented_data, list(stretch(data, factor = 1.2)))
    augmented_data <- c(augmented_data, list(stretch(data, factor = 0.8)))
    
    # Add noise
    noise <- rnorm(length(data), mean = 0, sd = sd(data) * 0.05)
    augmented_data <- c(augmented_data, list(data + noise))
  }
  
  return(unlist(augmented_data))
}

# Function to perform Fourier Transform and analyze frequencies
fourier_analysis <- function(sound_data, sample_rate) {
  N <- length(sound_data)
  fft_data <- fft(sound_data)
  fft_data <- fft_data[1:(N/2)]
  freqs <- (0:(N/2 - 1)) * (sample_rate / N)
  amplitude <- Mod(fft_data)
  return(data.frame(freqs = freqs, amplitude = amplitude))
}

# Define the model
create_rnn_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = input_shape, return_sequences = TRUE) %>%
    layer_lstm(units = 50) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam()
  )
  
  return(model)
}

# Train the model with validation and logging
train_model <- function(model, X_train, y_train, X_val, y_val, epochs = 10, batch_size = 32) {
  # tensorboard(log_dir = "logs")
  
  model %>% fit(
    X_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = list(X_val, y_val),
    callbacks = list(callback_tensorboard(log_dir = "logs"),
                     callback_model_checkpoint(filepath = "models/model-{epoch:02d}-{val_loss:.2f}.h5", save_best_only = TRUE))
  )
}

# Generate sound using the trained model
generate_sound <- function(model, input_sequence) {
  predicted_sound <- model %>% predict(input_sequence)
  return(predicted_sound)
}

# Split data into training, validation, and test sets
split_data <- function(sound_data, train_ratio = 0.7, val_ratio = 0.15) {
  n <- length(sound_data)
  train_index <- 1:(n * train_ratio)
  val_index <- (n * train_ratio + 1):(n * (train_ratio + val_ratio))
  test_index <- (n * (train_ratio + val_ratio) + 1):n
  
  list(
    train = sound_data[train_index],
    val = sound_data[val_index],
    test = sound_data[test_index]
  )
}

# Check if the model exists and load it or train a new one
model_file_path <- './models/music_model.h5'
if (file.exists(model_file_path)) {
  model <- load_model_hdf5(model_file_path)
  cat("Model loaded from file.\n")
} else {
  cat("Model file not found. Training a new model...\n")
  
  # Load and preprocess the recorded music data
  guitar_data <- load_recorded_data('./audio_files')
  
  sound_data <- guitar_data$data
  sample_rate <- guitar_data$sample_rate
  
  # Augment data
  augmented_data <- augment_data(list(sound_data))
  
  # Split data
  split <- split_data(augmented_data)
  X_train <- array_reshape(split$train[1:(length(split$train) - 1)], c(length(split$train) - 1, 1, 1))
  y_train <- array_reshape(split$train[2:length(split$train)], c(length(split$train) - 1, 1))
  
  X_val <- array_reshape(split$val[1:(length(split$val) - 1)], c(length(split$val) - 1, 1, 1))
  y_val <- array_reshape(split$val[2:length(split$val)], c(length(split$val) - 1, 1))
  
  # Define the input shape
  input_shape <- c(ncol(X_train), ncol(X_train))
  
  # Create the RNN model
  model <- create_rnn_model(input_shape)
  
  # Train the model with validation
  train_model(model, X_train, y_train, X_val, y_val)
  
  # Save the model to file
  dir.create('./models', showWarnings = FALSE)
  save_model_hdf5(model, model_file_path)
  cat("Model trained and saved to file.\n")
}

# Example usage
# Load and preprocess the sound data
recorded_data <- load_recorded_data('./audio_files')
sound_data <- recorded_data$data
sample_rate <- recorded_data$sample_rate

# Fourier analysis
fourier_data <- fourier_analysis(sound_data, sample_rate)

# Plot the Fourier analysis
plot(fourier_data$freqs, fourier_data$amplitude, type = 'l', col = 'blue', 
     xlab = 'Frequency (Hz)', ylab = 'Amplitude', 
     main = 'Fourier Analysis of Input Sound')

# Generate sound based on an input sequence
input_sequence <- array_reshape(sound_data[1:10], c(1, 10, 1))
generated_sound <- generate_sound(model, input_sequence)

# Save the generated sound to a file
generated_wave <- Wave(left = as.integer(generated_sound), samp.rate = sample_rate, bit = 16)
writeWave(generated_wave, 'generated_sound.wav')

# Perform Fourier analysis on the generated sound
generated_fourier_data <- fourier_analysis(generated_sound, sample_rate)

# Plot the Fourier analysis of the generated sound
plot(generated_fourier_data$freqs, generated_fourier_data$amplitude, type = 'l', col = 'red', 
     xlab = 'Frequency (Hz)', ylab = 'Amplitude', 
     main = 'Fourier Analysis of Generated Sound')

