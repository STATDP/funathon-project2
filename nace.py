import mlflow
from dotenv import load_dotenv
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder



load_dotenv(override=True)

df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")
ldf = str(len(df))
n_classes = str(df['code'].n_unique())

#for row in df.iter_rows(named=True):
#    print(row['code'])
#    print(row['name'])
#    print(row['label'])

#print(df.head(10))
#print("Antallet af rækker i filen: " + ldf)
#print("Antallet af forskellige NACE kode i filen: " + n_classes)

train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['code'])
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42, stratify=tmp_df['code'])

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())

all_codes  = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")

# Antal observationer pr. code i træningsdata
counts = train_df["code"].value_counts()

# Mindste antal
min_count = counts.min()

print(f"Mindste antal observationer for en code i train: {min_count}")

value_encoder = ValueEncoder(label_encoder=encoder)

print("Done")
