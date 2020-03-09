import pandas as pd
import numpy as np
from numpy.random import randn
np.random.seed(101)

df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
print(df)

# Selection and Indexing

# Selecting columns
# print(df['X'])
# print(df[['X', 'Z']])

# Selecting rows
# print(df.loc['A']) # based on labels
# print(df.iloc[2]) # based on position

# Selecting subsets of rows and columns
# print(df.loc['B','Y'])
# print(df.loc[['A','B'],['W','Y']])

# Creating new column
df['new'] = df['W'] + df['Y']
# print(df)

# Removing column
df.drop('new',axis=1, inplace=True) # if inplace is not specified, assign the DataFrame back to df # df = df.drop('new',axis=1)
# print(df)

# Removing rows
# df.drop('E',axis=0, inplace=True)
# print(df)

# Conditional Selection
# print(df[df>0])
# print(df[df['W']>0])
# print(df[df['W']>0]['Y'])
# print(df[df['W']>0][['Y','X']])

# For two conditions you can use | and & with parenthesis:
# print(df[(df['W']>0) & (df['Y'] > 1)])

# Resetting index
# Reset to default 0,1...n index
df.reset_index(inplace= True)
print(df)
df['States'] = 'CA NY WY OR CO'.split()
print(df)
df.set_index('States', inplace=True)
print(df)
df.drop('index', axis=1, inplace=True)
print(df)
