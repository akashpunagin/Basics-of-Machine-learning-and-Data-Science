import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
print(tips.head())

sns.distplot(tips['total_bill'])
# plt.show()
sns.distplot(tips['total_bill'],kde=False,bins=30)
# plt.show()

# jointplot allows you to basically match up two distplots for bivariate data. With your choice of what kind parameter to compare with:
# scatter reg resid kde hex
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
# plt.show()
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
# plt.show()
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
# plr.show()

# pairplot will plot pairwise relationships across an entire dataframe (for the numerical columns) and supports a color hue argument (for categorical columns).
sns.pairplot(tips)
# plt.show()
sns.pairplot(tips,hue='sex',palette='coolwarm')
# plt.show()

# rugplots are actually a very simple concept, they just draw a dash mark for every point on a univariate distribution. They are the building block of a KDE plot:
sns.rugplot(tips['total_bill'])
# plt.show()
