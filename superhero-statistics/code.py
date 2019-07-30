# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path


#Code starts here 

data = pd.read_csv(path)
#print(data)

data['Gender'].replace('-','Agender',inplace=True)
#rint(data)

gender_count= data['Gender'].value_counts()
print(gender_count)
gender_count.plot.bar()
plt.show()



# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
alignment.plot.pie()
plt.title('Character Alignment')
plt.show()



# --------------
#Code starts here
sc_df = data[['Strength','Combat']].copy()
#print(sc_df)
sc_covariance = sc_df['Strength'].cov(sc_df['Combat'])

#sc_strength= np.std(sc_df['Strength'], axis=0)
sc_strength= sc_df['Strength'].std()
print("sc_strength: ", sc_strength)
sc_combat= sc_df['Combat'].std()
print("sc_combat: ", sc_combat)

sc_pearson=sc_covariance/(sc_strength*sc_combat)
print("sc_pearson: ", sc_pearson)

ic_df = data[['Intelligence','Combat']].copy()
#print(ic_df)
ic_covariance = ic_df['Combat'].cov(ic_df['Intelligence'])

ic_intelligence= ic_df['Intelligence'].std()
print("ic_intelligence: ",ic_intelligence)
ic_combat= ic_df['Combat'].std()
print("ic_combat: ", ic_combat)

ic_pearson=ic_covariance/(ic_intelligence*ic_combat)
print("ic_pearson: ", ic_pearson)



# --------------
#Code starts here
total_high = np.quantile(data['Total'], 0.99)
print("total_high: ",total_high)

super_best= data[data['Total']>total_high]
print("super_best: \n\n",super_best)

super_best_names = [super_best['Name']]
print("\n\nsuper_best_names:\n\n",super_best_names)


# --------------
#Code starts here

fig = plt.figure()
ax_1 = fig.add_subplot(221)
ax_2 = fig.add_subplot(222, sharex=ax_1, sharey=ax_1)
ax_3 = fig.add_subplot(223, sharex=ax_1, sharey=ax_1)
#ax1.set_xlabel('xlabel')
ax_1.boxplot(super_best['Intelligence'])
ax_1.set_title('Intelligence')

ax_2.boxplot(super_best['Speed'])
ax_2.set_title('Speed')

ax_3.boxplot(super_best['Power'])
ax_3.set_title('Power')


