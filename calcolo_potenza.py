import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


FILE_PATH = '/content/drive/MyDrive/Colab Notebooks/Progetti /Riduttore/Dati riduttore  1.csv'

df = pd.read_csv(FILE_PATH)
df = df[df['Somma di Produzione ton'] > 1]
df = df.drop(columns=['Marca acciaio'])



######################################################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Use RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score # Use regression metrics



# Load the dataset
X = df[['Spessore richiesto [mm]','Media di Produttività R ton/h', 'Media di VELOCITA INGRESSO TUBO',
          'RIFERIMENTO INDUCTOTHERM', 'RIFERIMENTO ZONA 1 ASEA',
          'RIFERIMENTO ZONA 2 ASEA', 'RIFERIMENTO ZONA 3 ASEA',
          'Carbonio equivalente',]]
y = df[['Media di Somma Potenze elettriche kW']].values.ravel()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) #42

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=400, random_state=123)

# Fit the regressor on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model (using regression metrics)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


plt.figure(figsize=(8, 6))  # Set figure size (optional)
plt.scatter(y_test, y_pred, alpha=0.7)  # Create scatter plot
plt.xlabel("Actual Values (y_test)")  # Set x-axis label
plt.ylabel("Predicted Values (y_pred)")  # Set y-axis label
plt.title("Actual vs. Predicted Values")  # Set plot title
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)  # Add diagonal line
plt.grid(True)  # Add grid (optional)
plt.show()  # Display the plot


X2 = pd.concat([X, pd.DataFrame(y, columns=['y'])], axis=1)
X2 = X2.rename(columns={'y': 'Stima potenza elettrica assorbita'})


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load your dataset

X = X2
y = df[['Media di TEMPERATURA USCITA FORNO INDUCTOTHERM']].values.ravel()

X.columns = X.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_')
X.columns = X.columns.str.replace(' ', '_') # Replace spaces as well

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=400, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=400, random_state=42)
xgb_model.fit(X_train, y_train)

#Model deployment

rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)


combined_predictions = (rf_predictions + xgb_predictions) / 2

weight_rf = 0.5  # Weight for Random Forest
weight_xgb = 0.5  # Weight for XGBoost
combined_predictions = (weight_rf * rf_predictions + weight_xgb * xgb_predictions)


import numpy as np
from sklearn.linear_model import LinearRegression

# Stack predictions
stacked_predictions = np.column_stack((rf_predictions, xgb_predictions))

# Train a meta-model
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_test)

# Make final predictions
final_predictions = meta_model.predict(stacked_predictions)


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, final_predictions)
r2 = r2_score(y_test, final_predictions)
mape = np.mean(np.abs((y_test - final_predictions) / y_test)) * 100
mean_absolute_error = np.mean(np.abs(y_test - final_predictions))

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
print(f'Mean Absolute Error: {mean_absolute_error}')



import matplotlib.pyplot as plt

# Assuming final_predictions contains the predictions from the stacked model
# and y_test contains the actual target values

plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_predictions, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
plt.title('Actual vs. Predicted Values (Stacked Model)')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (Stacked Predictions) leveraging XGBoost and Random Tree')
plt.xlim(y_test.min(), y_test.max())
plt.ylim(y_test.min(), y_test.max())
plt.grid()
plt.show()


# Define input values for prediction

input_data = {
    'Spessore richiesto [mm]': [5],  # Example value, replace with your desired value
    'Media di Produttività R ton/h': [44.35],  # Example value, replace with your desired value
    'Media di VELOCITA INGRESSO TUBO': [60.58],  # Example value, replace with your desired value
    'RIFERIMENTO INDUCTOTHERM': [77],  # Example value, replace with your desired value
    'RIFERIMENTO ZONA 1 ASEA': [99],  # Example value, replace with your desired value
    'RIFERIMENTO ZONA 2 ASEA': [99],  # Example value, replace with your desired value
    'RIFERIMENTO ZONA 3 ASEA': [99],  # Example value, replace with your desired value
    'Carbonio equivalente': [0.54],  # Example value, replace with your desired value
}


# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Make predictions using the trained meta-model
predicted_power = rf_regressor.predict(input_df)

# Print the predicted temperature
print(f"Predicted Power: {predicted_power[0]:.2f} W")


input_data = {
    'Spessore_richiesto__mm_': [5],  # Example value, replace with your desired value
    'Media_di_Produttività_R_ton/h': [44.35],  # Example value, replace with your desired value
    'Media_di_VELOCITA_INGRESSO_TUBO': [60.58],  # Example value, replace with your desired value
    'RIFERIMENTO_INDUCTOTHERM': [77],  # Example value, replace with your desired value
    'RIFERIMENTO_ZONA_1_ASEA': [99],  # Example value, replace with your desired value
    'RIFERIMENTO_ZONA_2_ASEA': [99],  # Example value, replace with your desired value
    'RIFERIMENTO_ZONA_3_ASEA': [99],  # Example value, replace with your desired value
    'Carbonio_equivalente': [0.54],  # Example value, replace with your desired value
    'Stima_potenza_elettrica_assorbita': predicted_power # Example value, replace with your desired value
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Make predictions using the trained meta-model
predicted_temperature = meta_model.predict(
    np.column_stack((rf_model.predict(input_df), xgb_model.predict(input_df)))
)

# Print the predicted temperature
print(f"Predicted Temperature: {predicted_temperature[0]:.2f} °C")


