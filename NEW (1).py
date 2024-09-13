#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


df = pd.read_csv('SpotifySongs.csv')


# In[2]:


print(df.head())


# In[3]:


print(df.isnull().sum())


# In[4]:


features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
            'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms']


# In[5]:


X = df[features]
y = df['Popularity']


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[10]:


scaler = StandardScaler()
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])


# In[11]:


print(X_train.head())
print(y_train.head())


# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


model = LinearRegression()


# In[14]:


model.fit(X_train, y_train)


# In[15]:


y_pred = model.predict(X_test)


# In[16]:


y_pred


# In[31]:


new_song = {
    'Danceability': 0.9,
    'Energy': 0.65,
    'Key': 5,
    'Loudness': -6.0,
    'Mode': 1,
    'Speechiness': 0.04,
    'Acousticness': 0.3,
    'Instrumentalness': 0.6,
    'Liveness': 0.12,
    'Valence': 0.59,
    'Tempo': 140.0,
    'Duration_ms': 210000
}


# In[32]:


new_song_df = pd.DataFrame([new_song])


# In[33]:


new_song_scaled = scaler.transform(new_song_df[features])


# In[34]:


predicted_popularity = model.predict(new_song_scaled)


# In[37]:


print(f'Predicted Popularity: {predicted_popularity}')


# In[36]:


from sklearn.metrics import mean_squared_error, r2_score


# In[24]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# In[ ]:




