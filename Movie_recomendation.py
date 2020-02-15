#Import the three datasets

import pandas as pd 
import numpy as np

ratings = pd.read_csv('ratings.dat', sep = "::", names = ['UserID', 'MovieID', 'Rating' , 'Timestamp']) 
users = pd.read_csv('users.dat', sep = "::", names = ['UserID', 'Gender', 'Age', 'Occup ation', 'Zipcode']) 
movies = pd.read_csv('movies.dat', sep = "::", names = ['MovieID', 'Title', 'Genres'])

ratings.shape
users.shape
movies.shape
ratings.head()
users.head()
movies.head()
#Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age 
#Gender Occupation Rating
rating_user_Df = pd.merge(ratings,users,how='inner',on='UserID')
rating_user_Df.head()
rating_user_movies_Df = pd.merge(movies,rating_user_Df,how='inner',on='MovieID')
rating_user_movies_Df.shape
Master_Data = rating_user_movies_Df.drop(['Genres', 'Timestamp', 'Zipcode'], axis=1)
Master_Data.head()
Master_Data = rating_user_movies_Df.filter(['MovieID','Title','UserID','Age','Gender', 'Occupation','Rating'], axis=1)
Master_Data.head()

#User Age Distribution
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000 
Master_Data.plot(kind='scatter',x='UserID', y='Age')
Master_Data.plot(kind='kde',x='UserID', y='Age')
#User rating of the movie “Toy Story”
fltDF = Master_Data.loc[Master_Data.Title =='Toy Story (1995)'] 
fltDF

fltDF.plot(kind='kde',x='Title', y='Rating')
#Top 25 movies by viewership rating

topViewedMovies = Master_Data.groupby('Title').size().sort_values(ascending=False)[:25] 
topViewedMovies

topViewedMovies.plot()

#Find the ratings for all the movies reviewed by for a particular user of user id = 2696
movieRating = Master_Data.loc[Master_Data.UserID == 2696] 
movieRating 
movieRating.plot(kind='kde',x='UserID', y='Rating')

#Feature Engineering
#Find out all the unique genres

genre_list = set() 
for g in rating_user_movies_Df['Genres'].str.split('|'):    
    genre_list = genre_list.union(set(g)) 
genre_list

#Create a separate column for each genre category with a one-hot encoding ( 1 and 0) 
#whether or not the movie belongs to that genre
ind_min = rating_user_movies_Df.index.min() 
ind_min

ind_max = rating_user_movies_Df.index.max() 
ind_max

for g in genre_list:    
    rating_user_movies_Df[g] = 0     
rating_user_movies_Df[g]   

i = ind_min 
while i <= ind_max:    
    for g in rating_user_movies_Df.loc[i,'Genres'].split('|'):        
        rating_user_movies_Df.at[i,g] = 1        
        i += 1

rating_user_movies_Df.info()

rating_user_movies_Df.head()

rating_user_movies_Df.shape

#Determine the features affecting the ratings of any particular movie
rating_user_movies_Df.columns
from sklearn.preprocessing import LabelEncoder 
cat_var =rating_user_movies_Df.dtypes.loc[rating_user_movies_Df.dtypes == 'object'].index 
le =LabelEncoder() 
for var in cat_var:    
    rating_user_movies_Df[var] = le.fit_transform(rating_user_movies_Df[var])
rating_user_movies_Df.head()
depv = 'Rating' 
indepv = [x for x in rating_user_movies_Df.columns if x not in ['MovieID','Title','Genr es','UserID','Timestamp','Occupation','Zipcode',depv]]
rating_user_movies_Df[depv]
rating_user_movies_Df[indepv].head()

#Develop an appropriate model to predict the movie ratings

from sklearn.linear_model import LogisticRegression 
model = LogisticRegression() 
model.fit(rating_user_movies_Df[indepv],rating_user_movies_Df[depv])

model.predict(rating_user_movies_Df[indepv])
predicted_rating_user_movies_Df = model.predict(rating_user_movies_Df[indepv]) 
true_value = rating_user_movies_Df[depv]

predicted_rating_user_movies_Df

from sklearn.metrics import accuracy_score 
print("LogisticRegression Accuracy {:.2%}".format(accuracy_score(true_value,predicted_rating_user_movies_Df)))

#Random Forest
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth = 20, min_samples_leaf = 50, max_features = 'sqrt',n_estimators = 1000)
model.fit(rating_user_movies_Df[indepv],rating_user_movies_Df[depv])

predicted_rating_user_movies_Df = model.predict(rating_user_movies_Df[indepv])

predicted_rating_user_movies_Df
true_value = rating_user_movies_Df[depv]
true_value
print("Random Forest Accuracy {:.2%}".format(accuracy_score(true_value,predicted_rating_user_movies_Df)))

