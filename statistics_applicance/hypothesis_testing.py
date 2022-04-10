import pandas as pd
import numpy as np

data = pd.read_csv('data/student_performance.csv')
rename = {'race/ethnicity' : 'race',
                            'parental level of education': 'pareduc',
                            'test preparation course' : 'prepar',
                            'math score' : 'math',
                            'reading score' : 'reading',
                            'writing score' : 'writing'}
data = data.rename(columns=rename)
data = data.dropna()
data['gender'] = data['gender'].apply(lambda gender: 1 if gender == 'male' else 0)

tests = ['math', 'reading', 'writing']
# print(data.groupby(['race'])[tests].mean())
# I noticed that avg test score in all three tests (math, reading, writing)
# is increasing with race decreasing, i.e. group A has poorest results,
# and group E - highest.
race_options = sorted(data['race'].unique())
races = dict(zip(
    race_options,
    range(5, 0, -1)
))
# print(races)
data['race'] = data['race'].apply(lambda x: races[x])

# print(print(data.groupby(['pareduc'])[tests].mean()))
pareduc_options = data.groupby(['pareduc'])['math'].mean().sort_values().index
pareducs = dict(zip(
    pareduc_options,
    range(1, len(pareduc_options) + 1)
))
data['pareduc'] = data['pareduc'].apply(lambda x: pareducs[x])

# print(print(data.groupby(['lunch'])[tests].mean()))
data['lunch'] = data['lunch'].apply(lambda x: 1 if x == 'standard' else 0)

# print(data.groupby(['prepar'])[tests].mean())
data['prepar'] = data['prepar'].apply(lambda x: 1 if x == 'completed' else 0)

print(data.head())