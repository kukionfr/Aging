import numpy as np
import pandas as pd
import os
df = pd.DataFrame([],[],columns=[1,3,7,10,16,19,23,25,29,31])
print(df)
a=[1,2,3,4,5,6,7,8,9,10]
df.loc[os.path.join('asd','ds')]=a
print(df)

# for j in range(10):
#     a=[]
#     for i in range(10):
#         a.append(i)
#     print(a)
#     aa.append(a)
# print(aa)

