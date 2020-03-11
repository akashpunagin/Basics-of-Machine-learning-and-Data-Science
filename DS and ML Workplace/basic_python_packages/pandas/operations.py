import numpy as np
import pandas as pd

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abcdef','def','ghi','xyz']})
print(df)

# Unique Values
# print(df['col2'].unique()) # unique values in col2
# print(df['col2'].nunique()) # number of unuique values in col2
# print(df['col2'].value_counts())

# Selecting data
# Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]
# print(newdf)

# Applying Functions
def times2(x):
    return x*2
# print(df['col1'].apply(times2))
# print(df['col3'].apply(len))
# print(df['col1'].sum())

# Permanently Removing a Column
# del df['col1']
# print(df)

# Get column and index names:
# print(df.columns)
# print(df.index)


# Sorting and Ordering a DataFrame
# df.sort_values(by='col2', inplace=True)
# print(df)


# Find Null Values or Check for Null Values
print(df.isnull())

# Drop rows with NaN Values
print(df.dropna())
