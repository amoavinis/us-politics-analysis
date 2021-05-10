import pandas as pd

BASE_DIR = 'data/'
files = ['tweets-us-1.json', 'tweets-us-2.json', 'tweets-us-3.json', 'tweets-us-4.json']

df1 = pd.read_json(BASE_DIR+files[0])
df2 = pd.read_json(BASE_DIR+files[1])
df3 = pd.read_json(BASE_DIR+files[2])
df4 = pd.read_json(BASE_DIR+files[3])

merged_df = pd.concat([df1, df2, df3, df4])

merged_df.to_pickle('data/tweets-us-all.pkl')

print(merged_df.shape)
