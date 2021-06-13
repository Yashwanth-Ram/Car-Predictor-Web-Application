1. Created a new wnvironment with name carprediction and activated it
conda create -n carprediction python=3.7
conda activate carprediction

2. Install Jupyter notebooks if nor installed
pip install notebook
jupyter notebook

3.Creating and training the model

a.reading the csv file and browsing through it
import pandas as pd
df=pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
df.head()
df.shape
df.columns()
df.describe() --For boxplot analysis

b.check the unique values of categorical variables
print(df['Categorical_variable'].unique())

c.check for null values
df.isnull().sum()

d.Create a new derived feature as we cannot perform mathematical values on year column
final_dataset=df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type','transmission', 'owner']]
#Create a dummy variable with the max value
final_dataset["current_year"]=2020
#Subtract the max value by other values of year
final_dataset['no_year']=final_dataset['current_year']-final_dataset['year']
#Drop the dummy variable and year variable
final_dataset.drop(['year'], axis=1, inplace=True)
final_dataset.drop(['current_year'], axis=1, inplace=True)

e.Converting categorical variables into numerical using one hot encoding
final_dataset=pd.get_dummies(final_dataset, drop_first=True)

f.Check correlation between the variables
final_dataset.corr()

g.Use visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Use pairplot to view the distributions
sns.pairplot(final_dataset)

#Use Heat map to visulaize the correlations better
corrmat=final_dataset.corr()
top_corr_feature=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_feature].corr(),annot=True,cmap="RdYlGn")

h.Create dependent and independent features
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
X.head()
y.head()

i. Find the important features and drop the less important ones
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
#View the important feature
print(model.feature_importances_)
#Visualize the important features
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()

j.Train test split