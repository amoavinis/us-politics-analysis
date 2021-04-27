import pandas as pd
import preprocessor
import string


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


# Read the raw data
df = pd.read_csv('data/all_messages.csv')
# df = df.sample(n=20, random_state=0)

# 26782 are the examples of the minority class 1
# So randomly sample the same amount from class 0
df1 = df[df['Class'] == 1]
df0 = df[df['Class'] == 0].sample(26782, random_state=0)

df_total = pd.concat([df0, df1])

df_total['processed_msg'] = df_total['Message']\
    .apply(preprocessor.clean)\
    .str.lower()\
    .apply(remove_punctuations)

df_total.to_csv('data/processed-balanced.csv', index=False)

print("Dataset has been balanced and preprocessed.")
