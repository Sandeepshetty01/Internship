import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the advertising dataset
data = pd.read_csv("advertising.csv")

st.title('Advertising Data Analysis')
st.subheader('Dataset Information')

# Display the entire dataset with a scroll feature
st.write(data)

st.subheader('Descriptive Statistics')

# Display descriptive statistics
st.write(data.describe())

# Split the data into X (features) and Y (target)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=60)

# Model Training using Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)

# Display sliders for user input
st.sidebar.subheader('Enter Advertising Budget:')
tv_budget = st.sidebar.slider('TV Budget', min_value=0, max_value=300, value=150)
radio_budget = st.sidebar.slider('Radio Budget', min_value=0, max_value=50, value=25)
newspaper_budget = st.sidebar.slider('Newspaper Budget', min_value=0, max_value=100, value=50)

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'TV': [tv_budget],
    'Radio': [radio_budget],
    'Newspaper': [newspaper_budget]
})

# Predict sales based on user input
predicted_sales = model.predict(user_input)

# Calculate the accuracy of the model
y_pred = model.predict(x_test)
accuracy = r2_score(y_test, y_pred)

# Visualize the feature importance
st.subheader('Feature Importance Visualization')

# Calculate the coefficients for each feature
coefficients = model.coef_
feature_names = X.columns

# Create a bar chart to visualize feature importance
fig, ax = plt.subplots()
ax.barh(feature_names, coefficients)
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Importance')
st.pyplot(fig)

# Display the predicted sales and model accuracy
st.subheader('Predicted Sales Based on User Input:')
st.write(f'TV Budget: {tv_budget}, Radio Budget: {radio_budget}, Newspaper Budget: {newspaper_budget}')
st.write(f'Predicted Sales: {predicted_sales[0]:.2f}')

st.subheader('Model Accuracy:')
st.write(f'R-squared score: {accuracy:.2f}')
