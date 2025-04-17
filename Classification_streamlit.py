import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load and prepare the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model directly in the app
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Streamlit app interface
st.title('Iris Species Prediction')

# User inputs for features
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 4.1)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.3)
    return np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

input_df = user_input_features()

# Display the user input features
st.subheader('User Input Features')
st.write(input_df)

# Use the model to make a prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write(iris.target_names[prediction][0])
st.subheader('Prediction Probability')
st.bar_chart(prediction_proba.flatten())


# Scatter plot for Sepal length and width
fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
for i in range(len(iris.target_names)):
    ax.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=iris.target_names[i])
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.legend()
st.pyplot(fig)

# Add a new scatter plot for Petal length and width
fig2, ax2 = plt.subplots()
for i in range(len(iris.target_names)):
    ax2.scatter(X[y == i, 2], X[y == i, 3], c=colors[i], label=iris.target_names[i])
ax2.set_xlabel('Petal Length')
ax2.set_ylabel('Petal Width')
ax2.legend()

# Highlight the user's input on the plot
ax2.scatter(input_df[0, 2], input_df[0, 3], c='black', s=200, alpha=0.5, label='Your Input')
ax2.legend()

st.pyplot(fig2)
