'''
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
@st.cache
def load_data():
    df = pd.read_csv("O:\\Project ML\\city_day.csv")
    return df

df = load_data()

# Data Pre-processing
st.header("Data Pre-processing")
st.subheader("Original Data")
st.dataframe(df.head(10))

# Handle missing values
st.subheader("Handling Missing Values")
missing_values = df.isnull().sum()
st.write(missing_values)
df['PM2.5'] = df['PM2.5'].fillna(df['PM2.5'].median())
# Repeat the above line for other columns with missing values

st.subheader("Data after Handling Missing Values")
st.dataframe(df.head(10))

# Linear Regression Modeling
st.header("Linear Regression Modeling")

# Select features and target
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = 'AQI'

# Split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=2)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
st.subheader("Linear Regression Model Evaluation")
st.write(f"R-squared Score: {r2}")

# Streamlit App Configuration
st.title("Air Quality Prediction App")

# Show dataset
st.header("Original Dataset")
st.dataframe(df)

# Data Pre-processing
st.header("Data Pre-processing")
st.subheader("Handling Missing Values")
st.write("Handling missing values for PM2.5 column.")
st.dataframe(df.head(10))

# Linear Regression Modeling
st.header("Linear Regression Modeling")
st.subheader("Model Training and Evaluation")
st.write(f"R-squared Score: {r2}")

# Input for Prediction
st.header("Predict Air Quality Index (AQI)")
st.write("Enter values for the following features to predict AQI:")
input_values = {}
for feature in features:
    input_values[feature] = st.number_input(feature, value=df[feature].median())

# Prediction
prediction_button = st.button("Predict AQI")
if prediction_button:
    input_data = pd.DataFrame([input_values])
    prediction = lr.predict(input_data)[0]
    st.subheader(f"Predicted AQI: {prediction:.2f}")
'''
#-----------------------------------------------------------------------------#
'''
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv("O:/Project ML/city_day.csv")
    return df

# Data Pre-processing
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['PM2.5'] = df['PM2.5'].fillna(df['PM2.5'].median())
    # ... (filling other columns with median)
    return df

# Linear Regression Model
def train_linear_regression(Xtrain, Ytrain):
    lr = LinearRegression()
    lr.fit(Xtrain, Ytrain)
    return lr

# Random Forest Model
def train_random_forest(Xtrain, Ytrain):
    reg = RandomForestRegressor(max_depth=2, random_state=0)
    reg.fit(Xtrain, Ytrain)
    return reg

# Main Streamlit App
def main():
    st.title("Air Quality Analysis with Streamlit")

    # Load data
    df = load_data()

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df.head(10))

    columns_to_fillna = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2']
    for column in columns_to_fillna:
        df[column] = df[column].fillna(df[column].median())

    return df

    # Data Pre-processing
    st.subheader("Data Pre-processing")
    df_preprocessed = preprocess_data(df)
    st.dataframe(df_preprocessed.head(10))

    # Linear Regression Model
    st.subheader("Linear Regression Model")
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    labels = ['AQI']
    X = df_preprocessed[features]
    Y = df_preprocessed[labels]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=2)

    lr_model = train_linear_regression(Xtrain, Ytrain)
    lr_score = lr_model.score(Xtest, Ytest)
    st.write(f"Linear Regression R-squared Score: {lr_score:.4f}")

    # Random Forest Model
    st.subheader("Random Forest Model")
    rf_model = train_random_forest(Xtrain, Ytrain)
    rf_score = r2_score(Ytest, rf_model.predict(Xtest))
    st.write(f"Random Forest R-squared Score: {rf_score:.4f}")

if __name__ == "__main__":
    main()
'''
#--------------------------------------------------------------------------#
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv("O:/Project ML/city_day.csv")
    return df

# Data Pre-processing
def preprocess_data(df):
    df_copy = df.copy()  # Make a copy to avoid modifying the original data
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    
    # Handling null values
    columns_to_fillna = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2','O3','Benzene','Toluene','Xylene']
    for column in columns_to_fillna:
        df_copy[column] = df_copy[column].fillna(df_copy[column].median())

    df_copy = df_copy.dropna(subset=['AQI'])
    return df_copy

    

    

# Linear Regression Model
def train_linear_regression(Xtrain, Ytrain):
    lr = LinearRegression()
    lr.fit(Xtrain, Ytrain)
    return lr

# Random Forest Model
def train_random_forest(Xtrain, Ytrain):
    reg = RandomForestRegressor(max_depth=2, random_state=0)
    reg.fit(Xtrain, Ytrain)
    return reg

# Main Streamlit App
def main():
    st.title("Air Quality Analysis with Streamlit")

    # Load data
    df = load_data()

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df.head(10))

    # Data Pre-processing
    st.subheader("Data Pre-processing")
    df_preprocessed = preprocess_data(df)
    st.dataframe(df_preprocessed.head(10))

    # Linear Regression Model
    st.subheader("Linear Regression Model")
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    labels = ['AQI']
    X = df_preprocessed[features]
    Y = df_preprocessed[labels]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=2)

    lr_model = train_linear_regression(Xtrain, Ytrain)
    lr_score = lr_model.score(Xtest, Ytest)
    st.write(f"Linear Regression R-squared Score: {lr_score:.4f}")

    # Random Forest Model
    st.subheader("Random Forest Model")
    rf_model = train_random_forest(Xtrain, Ytrain)
    rf_score = r2_score(Ytest, rf_model.predict(Xtest))
    st.write(f"Random Forest R-squared Score: {rf_score:.4f}")

    #Heat Map
    st.subheader("Correlation Heatmap")
    df_corr = df_preprocessed[['Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
                               'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']].dropna().corr()

    plt.figure(figsize=(12, 8))
    sbn.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()
#---------------------------------------
    st.title("Mean AQI Values for Different Cities")
    # Calculate mean AQI for each city
    city_aqi_mean = df.groupby('City')['AQI'].mean()

    # Display mean AQI for each city
    st.subheader("Mean AQI Values for Different Cities")
    for city, mean_aqi in city_aqi_mean.items():
        st.write(f"{city}: {mean_aqi:.2f}")
#-------------------------------------------- 
    plt.figure(figsize=(12, 6))
    st.subheader("AQI Across Different Cities")
    sbn.barplot(x='City', y='AQI', data=df)
    plt.title('AQI Across Different Cities')
    plt.xlabel('City')
    plt.ylabel('AQI')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot()
#--------------------------------------------
    plt.figure(figsize=(12, 6))
    ax = sbn.scatterplot(x='Date', y='AQI', hue='AQI', data=df)
    ax.set_title('AQI Over Time')
    st.pyplot()

    
    st.title("Temporal Trends in AQI")
    # Generate line plot
    plt.figure(figsize=(12, 6))
    sbn.lineplot(x='Date', y='AQI', data=df)
    plt.title('Temporal Trends in AQI')
    plt.xlabel('Date')
    plt.ylabel('AQI')
#------------------ 3D ________plots
    st.title("3D Scatter Plot of AQI by Date and City")
    fig = px.scatter_3d(df, x="Date", y="AQI", z="City", color="City",
                        color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                        range_color=(-45, 225))
    st.plotly_chart(fig)
   

    fig = px.scatter_3d(df, x="Date", y="AQI", z="PM2.5", color="PM2.5",
                        color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                        range_color=(-45, 225))
    st.plotly_chart(fig)

    st.title("3D Scatter Plot of AQI by Date and Toluene")

    # Generate 3D scatter plot
    fig = px.scatter_3d(df, x="Date", y="AQI", z="Toluene", color="Toluene",
                        color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                        range_color=(-45, 225))
    
    # Display the plot
    st.plotly_chart(fig)

    st.title("3D Scatter Plot of AQI by Date and O3")

    # Generate 3D scatter plot
    fig = px.scatter_3d(df, x="Date", y="AQI", z="O3", color="O3",
                        color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                        range_color=(-45, 225))
    st.plotly_chart(fig)


    st.title("3D Scatter Plot of AQI by Date and CO")

    # Generate 3D scatter plot
    fig = px.scatter_3d(df, x="Date", y="AQI", z="CO", color="CO",
                        color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                        range_color=(-45, 225))
    st.plotly_chart(fig)

    fig = px.scatter_3d(df, x="Date", y="AQI",z="SO2", color="SO2",
                    color_continuous_scale=["#00FF00", "#FFC600", "#FF0060", "#B803BF"],
                    range_color=(-45, 225))
    st.plotly_chart(fig)
    fig = px.scatter_3d(df, x="Date", y="AQI",z="NH3", color="NH3",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
    st.plotly_chart(fig)
    fig = px.scatter_3d(df, x="Date", y="AQI",z="Benzene", color="Benzene",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
    st.plotly_chart(fig)

    fig = px.scatter_3d(df, x="Date", y="AQI", z="NO", color="NO",
                        color_continuous_scale=["#00FF00", "#FFC600", "#FF0060", "#B803BF"],
                        range_color=(-45, 225))
    st.plotly_chart(fig)
    fig = px.scatter_3d(df, x="Date", y="AQI",z="Xylene", color="Xylene",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
    st.plotly_chart(fig)

    # Display line plot using Streamlit's matplotlib integration
    st.title("Temporal Trends in AQI")
    st.pyplot(plt)

    st.title("Pollution Analysis")

    # Calculate mean AQI for each city
    city_aqi_mean = df.groupby('City')['AQI'].mean()

    # Display the least polluted cities with mean AQI less than 100
    st.subheader("Least Polluted Cities with Mean AQI less than 100:")
    least_polluted_cities = city_aqi_mean[city_aqi_mean < 100]
    for city, mean_aqi in least_polluted_cities.items():
        st.write(f"{city}: {mean_aqi:.2f}")

    # Identify the most and least polluted cities
    most_polluted_city = city_aqi_mean.idxmax()
    least_polluted_city = city_aqi_mean.idxmin()
    
    st.subheader("Most and Least Polluted Cities:")
    st.write(f"The most polluted city is: {most_polluted_city} with mean AQI: {city_aqi_mean[most_polluted_city]:.2f}")
    st.write(f"The least polluted city is: {least_polluted_city} with mean AQI: {city_aqi_mean[least_polluted_city]:.2f}")

    # Display most polluted cities with mean AQI greater than 200
    st.subheader("Most Polluted Cities with Mean AQI greater than 200:")
    most_polluted_cities = city_aqi_mean[city_aqi_mean > 200]
    for city, mean_aqi in most_polluted_cities.items():
        st.write(f"{city}: {mean_aqi:.2f}")
    #----------------------------------------------------------
    st.title("Aggregated Statistics of Hazardous Chemicals by City")

    # Extract relevant columns
    chemical_data = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]

    # Calculate aggregated statistics using describe()
    chemical_statistics_by_city = df.groupby('City')[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']].describe()

    # Display the aggregated statistics
    st.write(chemical_statistics_by_city)


if __name__ == "__main__":
    main()
