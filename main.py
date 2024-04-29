import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

penguins = pd.read_csv("penguins_size.csv")

penguins = penguins.dropna()
penguins.at[336, "sex"] = "FEMALE"

df = penguins.copy()
target = "species"
encode = ["sex", "island"]

# df.to_csv('penguins_clean.csv', index=False)

# Encoding the ordinal features
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) * 1
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}


def target_encode(val):
    return target_map[val]


df["species"] = df["species"].apply(target_encode)

# df.to_csv('penguins_clean.csv', index=False)
# print(df)
# print(df.columns)
# Features and Target
X = df.drop("species", axis=1)
y = df["species"]

# The model
clf = RandomForestClassifier()
clf.fit(X, y)

# Saving the model
pickle.dump(clf, open("penguins_clf.pkl", "wb"))
