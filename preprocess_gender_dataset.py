import preprocessor
import string

import pandas as pd

# Used for preprocessing
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# Read dataset
df = pd.read_csv("data/gender_classifier_dataset.csv")
print(df)

# Keep only gender, confidence, and text columns
df = df[["gender", "gender:confidence", "text"]]

# Print class counts
print(df["gender"].value_counts())
# female     6700
# male       6194
# brand      5942
# unknown    1117

# Keep only female and male
df = df[df["gender"].isin(["female", "male"])]

# Print annotation confidence
print(df["gender:confidence"].value_counts())
# 1.0000    13926
# 0.0000       71
# 0.6691       31
# 0.6591       30
# 0.6667       30
#           ...  
# 0.3553        1
# 0.3552        1
# 0.3393        1
# 0.6936        1
# 0.3381        1

# Keep only most condident annotations
df = df[df["gender:confidence"] == 1]

# Preprocess tweets
df['processed_text'] = df['text'].astype("str")\
    .apply(preprocessor.clean)\
    .str.lower()\
    .apply(remove_punctuations)

# Save preprocessed dataset
df.to_pickle("data/preprocessed_gender_classifier_dataset.pkl")
