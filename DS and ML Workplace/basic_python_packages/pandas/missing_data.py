import numpy as np
import pandas as pd

df = pd.DataFrame({'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]})
print(df)

# print(df.dropna()) # rows and columns containing NaN are deleted
# print(df.dropna(axis=0)) # only specified axis is deleted, axis=0 for rows, axis=1 for columns
# print(df.dropna(thresh=2)) # if NaN is below thresh value, rows / colums are not deleted
# print(df.fillna(value="Filled"))
print(df['A'].fillna(value=df['A'].mean()))
