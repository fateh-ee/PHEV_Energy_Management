import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch, Objective, HyperParameters

# Assuming data loading happens here

# Load your data
x1 = np.loadtxt('dc_J15_x_cor.txt')
y1 = np.loadtxt('dc_J15_y_cor.txt')
x2 = np.loadtxt('dc_unecelp_x_cor.txt')
y2 = np.loadtxt('dc_unecelp_y_cor.txt')

# Concatenate input and output data from different sources
input_data = np.concatenate([x1, x2], axis=0)
output_data = np.concatenate([y1, y2], axis=0)

input_data = np.round(input_data, 4)

# Shuffle the dataset
indices = np.arange(input_data.shape[0])
np.random.shuffle(indices)
input_data = input_data[indices]
output_data = output_data[indices]

# Convert to TensorFlow tensors
x_train = tf.constant(input_data, dtype=tf.float32)
y_train = tf.constant(output_data, dtype=tf.float32)

# Assuming the first 3 columns of y_train are for classification (one-hot encoded)
# and the rest are for regression
y_train_class = y_train[:, :3]  # Classification targets, adjust if necessary
y_train_reg = y_train[:, 3:]    # Regression targets

from tensorflow.keras.layers import Dense, Input, ELU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperParameters

def build_model(hp):
    inputs = Input(shape=(4,))
    x = inputs
    for i in range(hp.Int('num_layers', 1, 15)):  # Allowing up to 5 hidden layers
        x = Dense(units=hp.Int(f'units_layer_{i}', min_value=32, max_value=512, step=32))(x)
        # Adding ELU with a specified alpha after each Dense layer
        x = ELU(alpha=1.0)(x)

    # Assuming it's a single-label multi-class classification
    # Use 'softmax' for single-label classification, 'sigmoid' for multi-label classification
    class_output = Dense(3, activation='softmax', name='class_output')(x)  # Adjust according to your classification task
    reg_output = Dense(4, activation='linear', name='reg_output')(x)  # For regression

    model = Model(inputs=inputs, outputs=[class_output, reg_output])
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss={'class_output': 'categorical_crossentropy', 'reg_output': 'mean_squared_error'},  # Adjust loss for classification
                  metrics={'class_output': 'accuracy', 'reg_output': 'mean_squared_error'})
    return model





# Set up the tuner with corrected 'objective' argument
tuner = RandomSearch(
    build_model,
    objective=[Objective("class_output_accuracy", direction="max"), Objective("reg_output_mean_squared_error", direction="min")],
    max_trials=10,
    executions_per_trial=1,
    directory='my_tuning',
    project_name='multi_output_tuning'
)


from tensorflow.keras.callbacks import EarlyStopping

# Assuming you've defined 'build_model' and 'tuner' as before
tuner.search(x_train, {'class_output': y_train_class, 'reg_output': y_train_reg},
             epochs=100,  # Increased epochs
             validation_split=0.2,
             callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
             shuffle=True)  # Ensure data is shuffled


# Fetch the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model directly# Save the best model directl
best_model.save('tuner_class3.keras')  # Use '.ker