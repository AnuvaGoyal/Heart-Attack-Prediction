"""PLOTS

"""
#1
dataset.hist(figsize=(16,10))
plt.show()

#2
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(), annot=True)

#3
sns.countplot(dataset['output'])
plt.show()

#4
sns.displot(x= 'age', hue='output', data=dataset, alpha=0.6)
plt.show()

#5
attack = dataset[dataset['output']==1]
sns.displot(attack.age, kind = 'kde')
plt.show()

#6
sns.displot(attack.age, kind = 'ecdf')
plt.grid(True)
plt.show()

#7
ranges = [0, 30, 40, 50, 60, 70, np.inf]
labels = ['0-30', '30-40', '40-50', '50-60', '60-70', '70+']

attack['age'] = pd.cut(attack['age'], bins=ranges, labels=labels)
attack['age'].head()

sns.countplot(attack.age)
plt.show()

#8
fig, ax= plt.subplots(figsize=(8, 5))
sns.countplot(x= 'sex', hue='age', data=attack, ax=ax)
ax.set_xticklabels(['Female', 'Male'])
plt.legend(loc = 'upper right')
plt.show()
