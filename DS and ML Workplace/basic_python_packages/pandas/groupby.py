import pandas as pd

# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
print(df)

by_company = df.groupby('Company')
# print(by_company.mean())
# print(by_company.std())
# print(by_company.max())

# describe
print(by_company.describe())
# print(by_company.describe().transpose())
print(by_company.describe().transpose()['GOOG'])
