# Group member
# 13-Aditya Hendve
# 14-Akshar Hihoriya
# 15-Ashish Jha

# Dataset Name = realEstate dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from scipy import  stats

df = pd.read_csv("realEstate.csv")

columns = list(df.columns)
print(columns)

print(df.head()) 

new_df = df
#new_df.isnull()
#Checking for null values 
print(new_df.isnull().sum()) 
print("Missing values distribution: ")
print(new_df.isnull().mean())

print(new_df.duplicated().any()) 
print(new_df.duplicated())
print(new_df.shape)

new_df["status"].replace({"for_sale":"0", "sold":"1"}, inplace = True) 
print(new_df.head())
print(new_df.shape) 

print(new_df["state"].unique()) 

new_df['state'].replace({'Massachusetts':'0','Puerto Rico':'1','New York':'2','Maine':' 3','Connecticut':'4','Rhode Island':'5','Pennsylvania':'6','New Hampshire':'7'}, inplace = True)
print(new_df.head())
df2 = df

print(plt.figure(figsize = (10, 4), dpi = 100)) 
sns.boxplot(x = "bed", data = df2) 

plt.figure(figsize = (10, 4), dpi = 100) 
color_palette = sns.color_palette("Accent_r") 
sns.set_palette(color_palette)
sns.countplot(x = "state", data = df2) 

plt.figure(figsize = (10, 4), dpi = 100) 
color_palette = sns.color_palette("cool")
sns.set_palette(color_palette) 
sns.countplot(x = "status", data = df2) 

grp = dict(df2.groupby('state').groups) 
m = {}
for key, val in grp.items(): 
    if key in m:
        m[key] += len(val) 
    else:
        m[key] = len(val) 
        plt.title("Distribution of Region") 
        plt.pie(m.values(), labels = m.keys())

plt.figure(figsize = (10, 4), dpi = 100) 
color_palette = sns.color_palette("magma") 
sns.set_palette(color_palette)
sns.barplot(x = "state", y = "house_size", data = df2) 

data = pd.read_csv("realestate.csv")

data_numeric = data.drop(['status', 'city', 'state', 'zip_code'], axis=1)

plt.figure(figsize=(10, 6), dpi=100)
color_palette = sns.color_palette("magma")
sns.heatmap(data_numeric.corr(), vmax=0.9, annot=True, cmap=color_palette)
plt.title('Correlation Heatmap of Real Estate Data')
plt.show()

df1 = df2
print(plt.figure(figsize = (10, 4), dpi = 100)) 
sns.boxplot(x = "house_size", data = df1) 
percentile25 = df1['house_size'].quantile(0.25)
percentile75 = df1['house_size'].quantile(0.75)
iqr = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
df1[df1['house_size'] > upper_limit]
df1[df1['house_size'] < lower_limit]
new_df = df1[df1['house_size'] < upper_limit]
new_df.shape

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df1['house_size'])
plt.subplot(2,2,2)
sns.boxplot(df1['house_size'])
plt.subplot(2,2,3)
sns.distplot(new_df['house_size'])
plt.subplot(2,2,4)
sns.boxplot(new_df['house_size'])
plt.show()

warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df1['bath'])
plt.subplot(1,2,2)
sns.distplot(df1['house_size'])
plt.show()

print("Highest allowed",df1['house_size'].mean() + 3*df1['house_size'].std())
print("Lowest allowed",df1['house_size'].mean() - 3*df1['house_size'].std())

realEstate_dataset_copy = df2
mean_age = realEstate_dataset_copy['house_size'].mean()
std_age = realEstate_dataset_copy['house_size'].std()
realEstate_dataset_copy['house_size_normalized'] = (df2['house_size'] - mean_age) / std_age
print(realEstate_dataset_copy[['house_size', 'house_size_normalized']].head())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(realEstate_dataset_copy['house_size'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of house_size')
plt.xlabel('house_size')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist(realEstate_dataset_copy['house_size_normalized'], bins=20, color='green', alpha=0.7)
plt.title('Histogram of house_size (Normalized)')
plt.xlabel('Normalized house_size')
plt.ylabel('Frequency')
plt.tight_layout

import pandas as pd
median_bath = df['bed'].median()
df['bed'].fillna(median_bath, inplace=True)
print(df['bed'])
X = new_df.drop(columns=['status'])
y = new_df['status']
X_encoded = pd.get_dummies(X)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
print(pd.Series(y_resampled).value_counts())
plt.figure(figsize=(6,6))
sns.countplot(x=y_resampled)
plt.title('status class distribution after SMOTE')
plt.show()

x= df[['bed','bath','acre_lot', 'city', 'state', 'zip_code', 'house_size', 'price']]
y = df['status']
cross_val_score(SVC(),x, y, cv = 5)
cross_val_score(SVC(),x, y, cv = 5)
cross_val_score(DecisionTreeClassifier(), x, y, cv = 5)
cross_val_score(LogisticRegression(max_iter=5000), x, y, cv = 5)
cross_val_score(RandomForestClassifier(n_estimators=50), x, y, cv = 5)
cross_val_score(KNeighborsClassifier(),x, y ,cv = 5)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
model = RandomForestClassifier() 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, model.predict(X_test)))

dummy_dataset = df.copy()
print(dummy_dataset.head())
stats.ttest_1samp(data, 0) 
stats.ttest_1samp(data, 1)
data2 = df['price']
stats.ttest_1samp(data2, 0)

x= df[['bed','bath','acre_lot', 'city', 'state', ]]
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
model = LogisticRegression(max_iter=5000) 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, model.predict(X_test)))

X, y = make_classification(n_classes=2, class_sep=0.5,
weights=[0.05, 0.95], n_informative=2, n_redundant=0, flip_y=0,
n_features=2, n_clusters_per_class=1, n_samples=1000, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
model = KNeighborsClassifier() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, model.predict(X_test)))
feature_cols = ['bed','bath', 'city', 'state', 'zip_code','status',]
X = df[feature_cols] # Features
y = df.status
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

x = df.iloc[:,0:11] 
kmeans = KMeans(3) 
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['zip_code'],data_with_clusters['house_size'],c=data_with_clusters['price'],cmap='rainbow')