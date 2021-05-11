import preprocessor
import string
import pandas as pd

# Used for preprocessing
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

# Read dataset
df = pd.read_csv("data/user-age-dataset.csv")

# Print class counts
print(df["age"].value_counts())

# Preprocess tweets
df['text'] = df['text'].astype("str")\
    .str.lower()

# Save preprocessed dataset
df.to_csv("data/preprocessed-user-age-dataset.csv", index=False)
