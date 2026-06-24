import pickle
import pandas as pd

df = pd.read_pickle('springer_dataframe_26_categories.p')


# 1. Izbroj koliko puta se svaka klasa pojavljuje
klase_counts = df['LCSH_Label'].value_counts()

# 2. Filtriraj i uzmi samo one klase koje se pojavljuju 10 ili više puta
validne_klase = klase_counts[klase_counts >= 10].index

# 3. Zadrži u datasetu samo redove koji pripadaju tim validnim klasama
df_filtriran = df[df['LCSH_Label'].isin(validne_klase)].reset_index(drop=True)


print(df.shape)
print(df.head())
print(df.columns)
df.to_csv("springer_dataset1.csv", index=False)

counts = df['LCSH_Label'].value_counts().reset_index()
counts.columns = ['Klasa', 'Broj']
print(counts.to_string(index=False))

print(df_filtriran.shape)
print(df_filtriran.head())
print(df_filtriran.columns)
df_filtriran.to_csv("springer_dataset1.csv", index=False)

counts = df_filtriran['LCSH_Label'].value_counts().reset_index()
counts.columns = ['Klasa', 'Broj']
print(counts.to_string(index=False))