import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, PReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data

# driving cycle 1
x1 = np.loadtxt('dc_J15_x.txt') #input
y1 = np.loadtxt('dc_J15_y.txt') #output
# driving cycle 2
x2 = np.loadtxt('dc_unecelp_x.txt')
y2 = np.loadtxt('dc_unecelp_y.txt')
# driving cycle 3
x3 = np.loadtxt('dc_im240_x.txt')
y3 = np.loadtxt('dc_im240_y.txt')
# driving cycle 4
x4 = np.loadtxt('dc_unece_2_x.txt')
y4 = np.loadtxt('dc_unece_2_y.txt')
# driving cycle 5
x5 = np.loadtxt('dc_nycc_x.txt')
y5 = np.loadtxt('dc_nycc_y.txt')
# driving cycle 6
x6 = np.loadtxt('dc_sc03_x.txt')
y6 = np.loadtxt('dc_sc03_y.txt')

# concatenating all training dataset from 6 driving cycles
input_data = np.concatenate([x1, x2, x3, x4, x5, x6], axis=0) 
input_data = np.round(input_data, 4) # for input data simplification for DL model
output_data = np.concatenate([y1, y2, y3, y4, y5, y6], axis=0)

# Round and shuffle the data
indices = np.arange(input_data.shape[0])
np.random.shuffle(indices) # shuffling to get not affected by the order of data during training
input_data = input_data[indices]
output_data = output_data[indices]

# Split and convert to TensorFlow tensors
x_train, x_val, y_train, y_val = train_test_split(
    input_data, output_data, test_size=0.2, random_state=42) # taking 20 % of the total trianing data for validation testing, random state=42 is used to use the same dataset for training and validating during every time running the program

#
x_train, x_val = tf.constant(x_train, dtype=tf.float32), tf.constant(x_val, dtype=tf.float32) # making compatible to fetch the data into our neural network model
y_train, y_val = tf.constant(y_train, dtype=tf.float32), tf.constant(y_val, dtype=tf.float32)

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(4,))) # input has 4 features
    model.add(PReLU())  # using prelu activation function to the input layer
    

    for i in range(hp.Int('num_layers', 1, 15)): # varying the number of hidden layers between 1 and 15
        model.add(Dense(units=hp.Int(f'units_layer_{i}', min_value=8, max_value=256, step=8))) # nunber of neurons between 8 and 256
        model.add(PReLU()) # using prelu activation function

    
    
    model.add(Dense(7, activation='linear')) # output has 7 control actions
    
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error'])
    return model

early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=10, restore_best_weights=True) # it will stop training if the loss is not much improved for 10 epochs

tuner = RandomSearch(
    build_model,
    objective='val_mean_squared_error', # tuner will find out the best model based on this minimum loss function
    max_trials=50, # 50 random combinations will be created for tunning
    executions_per_trial=2, # every trial will execute 2 times for testing it's performance
    directory='auto_tuner', # saving directory
    project_name='keras_tuner_run'
)

tuner.search(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping]) # tuner will execute with 1000 epochs with early stopping at each random model  
best_model = tuner.get_best_models(num_models=1)[0] # top 1st model is achieved
history = best_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[early_stopping]) # again training it for observing model performance
best_model.save('model_tuned.keras') # saving
best_model.summary() # displaying the best model architecture

plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.title('Training and Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()

plt.show()
