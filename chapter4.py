import pandas as pd

df = pd.read_csv("./datafiles/KvsT.csv")

print(type(df["派閥"]))
print(type(df[["派閥", "体重"]]))