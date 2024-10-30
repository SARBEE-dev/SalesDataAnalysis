import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

st.write(""" # Analysis of loan dataset and Prediction""")
data = pd.read_csv('loandata/credit_risk_dataset.csv')
st.subheader("Credit risk  Data")
st.write(data)
st.subheader("Selecting filtered columns")
columns = st.multiselect("Columns:",data.columns)
filter = st.radio("Choose by:", ("inclusion","exclusion"))

if filter == "exclusion":
    columns = [col for col in data.columns if col not in columns]
data[columns]

st.subheader("Descriptive Statistics")
st.write(data.describe().T)
st.subheader("checking the null values in each of the columns")
st.write(data.isnull().sum())
st.subheader("Analysis of categorical and numerical variables")
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
st.write(f'categorical_columns are: {categorical_columns}')
st.write('\n')
st.write(f'numerical_columns are: {numerical_columns}')
st.subheader("dropping the nan values of the person_emp_length and loan_int_rate column and displaying "
        "the null values again to check")
data.dropna(subset=['person_emp_length', 'loan_int_rate'], inplace=True)
st.write(data.isnull().sum())
fig, ax = plt.subplots()
data.hist(bins=8, column="loan_amnt", grid=False, figsize=(8, 8), color="#86bf91", zorder=2, rwidth=0.9, ax=ax,)
st.write(fig)

st.subheader("Cleaned descriptive statistics")
st.write(data.describe().T)



columns = data.columns.tolist()
selected_columns = st.multiselect("select column", columns, default="loan_intent")
s = data[selected_columns[0]].str.strip().value_counts()
trace = go.Bar(x=s.index,y=s.values,showlegend = True)
layout = go.Layout(title = "test")
data = [trace]
fig = go.Figure(data=data,layout=layout)
st.plotly_chart(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('loandata/credit_risk_dataset.csv')
    return df


df = load_data()

# Sidebar for user input
st.sidebar.header("Enter input for loan prediction")


def user_input_features():
    person_age = st.sidebar.slider("Age", 18, 100, 30)
    person_income = st.sidebar.number_input("Annual Income", 1000, 500000, 50000)
    person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    person_emp_length = st.sidebar.slider("Employment Length (months)", 0, 600, 60)
    loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT",
                                                       "DEBTCONSOLIDATION"])
    loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.sidebar.number_input("Loan Amount", 100, 50000, 5000)
    loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", 0.0, 30.0, 10.0)
    loan_percent_income = st.sidebar.slider("Loan % of Income", 0.01, 1.0, 0.2)
    cb_person_default_on_file = st.sidebar.selectbox("Default on File", ["Y", "N"])
    cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (years)", 0, 30, 5)

    data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }
    return pd.DataFrame(data, index=[0])


user_input = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(user_input)


# Data preprocessing
def preprocess_data(df):
    # Handle missing values
    df['person_emp_length'].fillna(df['person_emp_length'].mean(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace=True)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    df['person_home_ownership'] = label_encoder.fit_transform(df['person_home_ownership'])
    df['loan_intent'] = label_encoder.fit_transform(df['loan_intent'])
    df['loan_grade'] = label_encoder.fit_transform(df['loan_grade'])
    df['cb_person_default_on_file'] = label_encoder.fit_transform(df['cb_person_default_on_file'])

    return df


df = preprocess_data(df)


# Preprocess user input (same encoding)
def preprocess_user_input(user_input):
    label_encoder = LabelEncoder()
    user_input['person_home_ownership'] = label_encoder.fit_transform(user_input['person_home_ownership'])
    user_input['loan_intent'] = label_encoder.fit_transform(user_input['loan_intent'])
    user_input['loan_grade'] = label_encoder.fit_transform(user_input['loan_grade'])
    user_input['cb_person_default_on_file'] = label_encoder.fit_transform(user_input['cb_person_default_on_file'])

    return user_input


user_input = preprocess_user_input(user_input)

# Visualizations
st.subheader('Data Analysis Visualizations')

# Age distribution
st.write("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df['person_age'], kde=True, ax=ax)
st.pyplot(fig)

# Income distribution
st.write("Income Distribution")
fig, ax = plt.subplots()
sns.histplot(df['person_income'], kde=True, ax=ax)
st.pyplot(fig)

# Loan amount distribution
st.write("Loan Amount Distribution")
fig, ax = plt.subplots()
sns.histplot(df['loan_amnt'], kde=True, ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Boxplot of loan amount by loan status
st.write("Boxplot of Loan Amount by Loan Status")
fig, ax = plt.subplots()
sns.boxplot(x='loan_status', y='loan_amnt', data=df, ax=ax)
st.pyplot(fig)

# Countplot of home ownership
st.write("Countplot of Home Ownership")
fig, ax = plt.subplots()
sns.countplot(x='person_home_ownership', data=df, ax=ax)
st.pyplot(fig)

# Machine Learning - Logistic Regression
st.subheader('Loan Status Prediction using Logistic Regression')

# Feature and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Save the model
joblib.dump(logistic_model, 'logistic_model.joblib')

# Load the model
model = joblib.load('logistic_model.joblib')

# Apply the model to user input
st.subheader('Prediction')
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

st.write(f"Loan Status Prediction: {'Default' if prediction[0] == 1 else 'Paid'}")
st.write(f"Prediction Probability: {prediction_proba}")
