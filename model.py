import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend(shadow=True)
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results', fontsize=14)
    plt.xlabel('X axis values', fontsize=11)
    plt.ylabel('Y axis values', fontsize=11)
    plt.savefig('model_results.png', dpi=120)
    plt.show()  # Show plot as well

def mae(y_test, y_pred):
    return tf.reduce_mean(tf.abs(y_test - y_pred))

def mse(y_test, y_pred):
    mse_metric = tf.keras.metrics.MeanSquaredError()  # Instantiate the metric
    mse_metric.update_state(y_test, y_pred)  # Update the state
    return mse_metric.result().numpy()
    
# Check Tensorflow version
print(tf.__version__)

# Create features and labels
X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

# Split data into train and test sets
X_train = X[:40].reshape(-1, 1)
y_train = y[:40].reshape(-1, 1)
X_test = X[40:].reshape(-1, 1)
y_test = y[40:].reshape(-1, 1)

# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),  # Increased units and added activation
    tf.keras.layers.Dense(32, activation='relu'),  # Additional hidden layer
    tf.keras.layers.Dense(1) 
])

# Compile the model
model.compile(loss='mae', optimizer=tf.keras.optimizers.SGD(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Fit the model
model.fit(X_train, y_train, epochs=100)

# Make and plot predictions for model
y_preds = model.predict(X_test)

# Print the shape of y_preds after defining it
print(y_preds.shape)

plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Calculate model metrics
mse_value = mse(y_test, y_preds.squeeze())
mae_value = np.round(float(mae(y_test, y_preds.squeeze())), 2)
mse_1 = np.round(float(mse_value), 2)  # Use mse_value for conversion to float
print(f'\nMean Absolute Error = {mae_value}, Mean Squared Error = {mse_1}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'Mean Absolute Error = {mae_value}, Mean Squared Error = {mse_1}.')

# Save the model
model.save('my_model.keras')
