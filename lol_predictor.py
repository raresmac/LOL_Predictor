from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import db_retriever

def check_x(x):
    length = len(x[0])
    l = len(x)
    i = 0
    while i < l:
        if len(x[i]) != length:
            print(i)
            x.pop(i)
            y.pop(i)
            l -= 1
        i += 1

X, y = db_retriever.get_db('bot')
print(len(X))
check_x(X)

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define input dimension
input_dim = X_train.shape[1]

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, input_shape=(input_dim,), activation='relu'),
    tf.keras.layers.Dropout(0.1),  # Add dropout regularization
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.1),  # Add dropout regularization
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Binary cross-entropy loss for binary classification
              metrics=['accuracy'])
# K.set_value(model.optimizer.learning_rate, 0.01)

# Train the model
model.fit(X_train, y_train, batch_size=500, epochs=20)
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

# # Make predictions
# predictions = model.predict(X_test)
# binary_predictions = (predictions > 0.5).astype(int)  # Apply threshold of 0.5 for binary classification
# def testing():
#     wrong = 0
#     for i in range(len(binary_predictions)):
#         if y_test[i] != binary_predictions[i]:
#             wrong += 1
#     print(wrong)
#     return 1 - wrong / len(predictions)
#
# print("Accuracy: ", testing())
