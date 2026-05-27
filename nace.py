import mlflow
from dotenv import load_dotenv
import polars as pl

load_dotenv(override=True)

df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")
ldf = str(len(df))
n_classes = str(df['code'].n_unique())

print(df.head(10))
print("Antallet af rækker i filen: " + ldf)
print("Antallet af forskellige NACE kode i filen: " + n_classes)

print("Done")