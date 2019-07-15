# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
#print(data['Rating'])
data.hist(column='Rating')

data = data[data['Rating'] <= 5]

data.hist(column = 'Rating')
#Code ends here


# --------------
# code starts here

total_null = data.isnull().sum()

percent_null = (total_null/data.isnull().count())

missing_data = pd.concat((total_null,percent_null),axis=1,keys=['Total','Percent']) 

print(missing_data)

data1 = data.dropna()

total_null_1 = data1.isnull().sum()

percent_null_1 = (total_null_1/data1.isnull().count())

missing_data_1 = pd.concat((total_null_1,percent_null_1),axis=1,keys=['Total','Percent']) 

print(missing_data_1)

# code ends here


# --------------

#Code starts here

sns.catplot(x="Category", y="Rating", kind="box", data=data,height=10)
plt.title('Rating vs Category [BoxPlot]')
plt.xticks(rotation=90)

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())
#print(data)
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].astype(int)
#print(data['Installs'])
le = LabelEncoder()
le.fit(data['Installs'])
data['Installs'] = le.transform(data['Installs'])

sns.regplot(x="Installs",y="Rating",data=data)
plt.title('Rating vs Installs [RegPlot]')
#Code ends here



# --------------
#Code starts here

print(data['Price'].value_counts)

data["Price"] = data["Price"].str.replace("$","").astype(float)

sns.regplot(x="Price",y="Rating",data=data)
plt.title('Rating vs Price [RegPlot]')
#Code ends here


# --------------

#Code starts here

#print(data['Genres'].unique())


data['Genres'] = data['Genres'].str.split(';').str[0]
#print(data['Genres'])

#gr_mean = data.groupby(['Genres','Rating'], as_index=False).mean()
gr_mean = data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()
#print(gr_mean)
#gr_mean = data.groupby(['Genres', 'Rating'],as_index=False).mean()

#print(gr_mean)
gr_mean.describe()

gr_mean =  gr_mean.sort_values(by='Rating',axis = 0)
print(gr_mean.head(1))
#DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#print(gr_mean.head())

#Code ends here


# --------------

#Code starts here

#data['Last Updated'] = data['Last Updated'].dt.strftime('%m/%d/%Y')

data['Last Updated'] = pd.to_datetime(data['Last Updated'], errors='coerce')
#print(data['Last Updated'])

max_date = data['Last Updated'].max()
#print(max_date)

data['Last Updated Days'] = max_date - data['Last Updated']

data['Last Updated Days'] = data['Last Updated Days'].dt.days
#print(data['Last Updated Days'])

sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title('Rating vs Last Updated [RegPlot]')

#Code ends here


