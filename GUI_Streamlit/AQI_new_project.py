#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sbn
from sklearn import model_selection


# In[5]:


#df = pd.read_csv()
df = pd.read_csv("O:\Project ML\city_day.csv")
df.head(10)


# In[6]:


df.tail(10)


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df['Date'] = pd.to_datetime(df['Date'])
df.info()


# **Data Pre-processing**
# -dealing with missing data

# In[11]:


df.isnull().sum()


# In[12]:


df['PM2.5']=df['PM2.5'].fillna((df['PM2.5'].median()))
df['PM10']=df['PM10'].fillna((df['PM10'].median()))
df['NO']=df['NO'].fillna((df['NO'].median()))
df['NO2']=df['NO2'].fillna((df['NO2'].median()))
df['NOx']=df['NOx'].fillna((df['NOx'].median()))
df['NH3']=df['NH3'].fillna((df['NH3'].median()))
df['CO']=df['CO'].fillna((df['CO'].median()))
df['SO2']=df['SO2'].fillna((df['SO2'].median()))


# In[13]:


df['O3']=df['O3'].fillna((df['O3'].median()))
df['Benzene']=df['Benzene'].fillna((df['Benzene'].median()))
df['Toluene']=df['Toluene'].fillna((df['Toluene'].median()))
df['Xylene']=df['Xylene'].fillna((df['Xylene'].median()))
df['AQI']=df['AQI'].fillna((df['AQI'].median()))
df['AQI_Bucket']=df['AQI_Bucket'].fillna('Moderate')
df = df


# In[14]:


df.isna().sum()


# In[15]:


df.tail(15)


# In[16]:


df['City'].value_counts()


# In[17]:


df.head(15)


# In[ ]:





# Data Collection:gathering a comprehensive dataset that contains crucial information for our Air Quality Index prediction task.
# 
# Data Analysis: Dive deep into the data to gain valuable insights and a better understanding of the patterns and trends within it.
# 
# Data Preprocessing: to prepare the data for model training by handling missing values, outliers, and ensuring its suitability for Machine Learning algorithms.

# In[ ]:





# In[ ]:





# In[18]:


df['Date'] = pd.to_datetime(df['Date'])


# Classification: It predicts the class of the dataset based on the independent input variable. 
# 
# Regression: It predicts the continuous output variables based on the independent input variable.

# ***Lineear Regression***

# In[19]:


hazardous_chemicals = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]
labels = df['AQI']


# In[20]:


hazardous_chemicals.head(5)
labels.head(5)


# **Splitting data**

# In[21]:


#splitting into train & test

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(hazardous_chemicals,labels,test_size=0.2, random_state=2)


# In[22]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import classification_report
from sklearn import metrics
reg = RandomForestRegressor(max_depth = 2, random_state=0)
reg.fit(Xtrain,Ytrain)
print(reg.predict(Xtest))


# In[23]:


y_pred = reg.predict(Xtest)
from sklearn.metrics import r2_score


# In[24]:


r2_score(Ytest,y_pred)  #accuracy qiute less ,so replace null columns with their median


# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lr = LinearRegression()
lr.fit(Xtrain,Ytrain)


# In[31]:


lr.score(Xtest,Ytest)


# In[32]:


y_pred = lr.predict(Xtest)
y_pred


# ***Creating Linear Regression Model***

# In[33]:


#a=model_selection.coef_   #This attribute of the linear regression model (model) represents the coefficients (weights) associated with each feature (independent variable)
#b= model.intercept_   #This attribute represents the intercept (y-intercept) of the linear regression line.
#y_pred=a*x+b
#y_pred


# In[34]:


x = df.iloc[:,2:13].values
y = df.iloc[:,-2].values


# In[35]:


x


# In[36]:


y


# ***result prediction***

# In[37]:


import seaborn as sns
from sklearn import linear_model

model=linear_model.LinearRegression()
model.fit(x,y)


# ***Linear Regression is a fundamental machine learning model used for predicting a continuous outcome variable (also called the dependent variable) based on one or more predictor variables (independent variables). It assumes a linear relationship between the input features and the target variable.

# In[38]:


model.predict([[67.450578,118.127103,0.97,15.69,16.46,23.483476,0.97,24.55,34.06,3.68000,5.500000]])


# In[39]:


model.predict([[60.450578,120.127103,1.00,15.00,16.50,20.483476,1.00,25.55,35.06,2.68000,5.500000]])


# In[ ]:





# In[40]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from pandas_visual_analysis import VisualAnalysis


# In[41]:


sns.displot(df['NO'],aspect=3,height=5)
sns.displot(df['NOx'],aspect=3,height=5)
sns.displot(df['NO2'],aspect=3,height=5)


# In[42]:


sns.displot(df['NO2'],aspect=3,height=5)


# ***Finding values from same cities***

# In[43]:


import plotly.express as px  #Plotly Express is often used for quick and easy generation of various charts and visualizations with a concise syntax.
import plotly.graph_objects as go  #provides a more granular level of control over the appearance
from plotly.subplots import make_subplots 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[44]:


target_city = 'Ahmedabad'
City_air = df[df['City'] == target_city ]
City_air


# **AQI  over a time period from 2015-2020**

# In[45]:


ax = sns.scatterplot(x='Date',
                y='AQI',
                hue= 'AQI',
                data=df
                    )
ax.set_title('AQI over time')
plt.show()


# In[61]:


df3 = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]
print('Distribution of different pollutants between 2015-2020')
df3.plot(kind='line',figsize=(16,16),cmap='coolwarm',subplots=True,fontsize=10);


# In[46]:


#plotting the bubble chart
fig = px.scatter(df, x='City', y='AQI')
fig.show()


# In[54]:


df.columns


# In[52]:


sns.pairplot(df,
             vars=['Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI'],)
#             hue='Type_Main')
plt.show()
#Scatterplots show the relationships between pairs of variables. 
#Patterns in scatterplots can indicate correlations or trends in the data.


# In[53]:


df_corr = df[['Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
       'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']].dropna().corr()

#The corr() method calculates the pairwise correlation coefficients between the selected columns. 


# In[55]:


sns.heatmap(df_corr, annot=True)


# In[47]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="CO", color="CO",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()
#Plotly Express graph


# In[48]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="PM2.5", color="PM2.5",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()
#The plot allows you to explore the relationships between 
#Since Plotly Express generates interactive plots, you can hover over data points to see specific values, rotate the plot, and zoom in/out for a more detailed examination.


# In[49]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="NH3", color="NH3",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()


# In[59]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="NO", color="NO",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()


# In[50]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="O3", color="O3",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()


# In[61]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="SO2", color="SO2",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()


# In[64]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="Toluene", color="Toluene",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()


# In[62]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="Benzene", color="Benzene",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()


# In[65]:


len(df.query("City in ['Chennai']"))


# In[67]:


#print(df['City'].value_counts)
Ahmedabad_count = df.query("City == 'Ahmedabad'") 
Ahmedabad_count
Ahm_df = df.iloc[:2,14].values

#y = df.iloc[:,-2].values


# In[ ]:





# In[68]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="City", color="City",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-55, 225))
fig.show()


# In[69]:


#ahm_df = df.iloc[:,-2:].query("City=='Ahmedabad'")
ahm_df = df.query("City=='Ahmedabad'").iloc[:,[0,1,-2,-1]]
aiz_df = df.query("City=='Aizawl'").iloc[:,[0,1,-2,-1]]

# In[71]:


ahm_df


# In[70]:


aiz_df


# In[ ]:





# In[83]:


print(df.iloc[:,:0])


# In[93]:


custome_bins = [0,100,200,300,400]
sns.histplot(df['AQI'],bins=custome_bins,kde=True)
plt.title('Distribution of AQI values')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.show()


# ***AQI insight yearly***

# In[133]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['AQI'].plot(figsize=(12, 6), title='AQI Over Time')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.show()


# In[105]:


df.tail()


# In[110]:


ahm_df.head()


# In[ ]:


### AQI value city-wise


# In[33]:


plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='AQI', data=df)
plt.title('AQI Across Different Cities')
plt.xlabel('City')
plt.ylabel('AQI')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[78]:


print(ahm_df['AQI'].mean())
print(aiz_df['AQI'].mean())
print(df[df['City'] == 'Amaravati']['AQI'].mean())
print(df[df['City'] == 'Amritsar']['AQI'].mean())
print(df[df['City'] == 'Bengaluru']['AQI'].mean())
print(df[df['City'] == 'Bhopal']['AQI'].mean())
print(df[df['City'] == 'Chandigarh']['AQI'].mean())
print(df[df['City'] == 'Chennai']['AQI'].mean())
print(df[df['City'] == 'Coimbatore']['AQI'].mean())
print(df[df['City'] == 'Delhi']['AQI'].mean())
print(df[df['City'] == 'Ernakulam']['AQI'].mean())
print(df[df['City'] == 'Gurugram']['AQI'].mean())
print(df[df['City'] == 'Guwahati']['AQI'].mean())
print(df[df['City'] == 'Hyderabad']['AQI'].mean())
print(df[df['City'] == 'Jaipur']['AQI'].mean())
print(df[df['City'] == 'Kochi']['AQI'].mean())
print(df[df['City'] == 'Kolkata']['AQI'].mean())
print(df[df['City'] == 'Lucknow']['AQI'].mean())
print(df[df['City'] == 'Mumbai']['AQI'].mean())
print(df[df['City'] == 'Patna']['AQI'].mean())
print(df[df['City'] == 'Shillong']['AQI'].mean())
print(df[df['City'] == 'Visakhapatnam']['AQI'].mean())


# ##plotted  mean AQI of all the cities

# In[87]:


'''import plotly.express as px
fig1 = px.scatter(df,x="PM10",y="AQI")
fig1.show()
fig2 = px.scatter(df,x="NO",y="AQI")
fig2.show() '''


# In[85]:


city_aqi_mean = df.groupby('City')['AQI'].mean()
least_polluted_cities = city_aqi_mean[city_aqi_mean < 100]

# Display the least polluted cities with mean AQI less than 100
print("Least polluted cities with mean AQI less than 100:")
for city, mean_aqi in least_polluted_cities.items():
    print(f"{city}: {mean_aqi:.2f}")


# In[83]:


# Calculate the mean AQI for each city
city_aqi_mean = df.groupby('City')['AQI'].mean()

# Identify the most and least polluted cities
most_polluted_city = city_aqi_mean.idxmax()
least_polluted_city = city_aqi_mean.idxmin()


# In[84]:


print(f"The most polluted city is: {most_polluted_city} with mean AQI: {city_aqi_mean[most_polluted_city]:.2f}")
print(f"The least polluted city is: {least_polluted_city} with mean AQI: {city_aqi_mean[least_polluted_city]:.2f}")


# In[91]:


most_polluted_cities = city_aqi_mean[city_aqi_mean > 200]

print("Most polluted cities with mean AQI greater than 200:")
for city, mean_aqi in most_polluted_cities.items():
    print(f"{city}: {mean_aqi:.2f}")


# In[48]:


df.info()


# In[ ]:


### Temporal trends in AQI over a Time


# In[40]:


plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='AQI', data=df)
plt.title('Temporal Trends in AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.show()


# In[81]:


fig = px.scatter_3d(df, x="Date", y="AQI",z="City", color="City",
                    color_continuous_scale=["#00FF00", "#FFC800", "#FF0000", "#B803BF"],
                    range_color=(-45, 225))
fig.show()


# In[94]:


# Identify hazardous chemicals
hazardous_chemicals = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Explore chemical distribution
chemical_distribution = df[hazardous_chemicals]
chemical_distribution.describe()

correlation_matrix = df[hazardous_chemicals + ['AQI']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Matrix of Hazardous Chemicals and AQI')
plt.show()



# #### Representation of hazardous_chemicals present in each city ##

# In[95]:


# Extract relevant columns
chemical_data = df[hazardous_chemicals]

# Calculate aggregated statistics using describe()
#chemical_statistics = chemical_data.describe().transpose()

chemical_statistics_by_state = df.groupby('City')[hazardous_chemicals].describe()

# Display the aggregated statistics
print(chemical_statistics_by_state)


# In[46]:


chemical_means = chemical_data.mean()
print(chemical_means)


# In[ ]:




