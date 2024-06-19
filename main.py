import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Read the Excel file
file_path = "/Users/javiergarcia/MANADINE/3 SEMESTER/ACTUARIAL/Tarea RD/tablaOlhsson.xlsx"
df = pd.read_excel(file_path)

# Display basic information about the dataset
print("Informacion del dataset:")
print(df.info())

# Display the first few rows of the dataset
print("\n Vista inicial:")
print(df.head())

# Descriptive statistics
print("\n Analisis descriptivo:")
print(df.describe())

# Correlation matrix
correlation_matrix = df.corr()
print("\n matriz de correlación:")
print(correlation_matrix)

# Heatmap for visualizing the correlation matrix
plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", 
            fmt=".2f", linewidths=.5, annot_kws={"size": 7})
plt.title("Matriz de correlación")

# Show the plots
plt.show()

# replace severidad and freq null with 0
df.fillna({'severidad': 0,'freq.sin': 0}, inplace=True)

# Create dummy variables
df = pd.get_dummies(df, columns=['zona', 'clase veh', 'Bonus'])

# Print info
print("Informacion del dataset:")
print(df.info())

# Descriptive statistics
print("\nVerificacion de limpieza:")
print(df.describe())

# Separate features and target variable
X = df.drop('severidad', axis=1)
y = df['severidad']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', 
                input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, 
        validation_split=0.2, verbose=0)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the error or any other performance metric you are interested in
error = np.sqrt(np.mean(np.square(y_pred.flatten() - y_test)))

print(f'Root Mean Squared Error (RMSE): {error}')
