import csv
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("open/train_data.csv", engine="python", error_bad_lines=False)
nan_list = [27556, 34092, 34201, 66897, 133273, 216203]
data = data[~data["id"].isin(nan_list)].reset_index(drop=True)

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

with open("open/train_gboot.tsv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["text", "summary"])
    for idx, row in train_data.iterrows():
        writer.writerow([row["text"], row["summary"]])

with open("open/val_gboot.tsv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["text", "summary"])
    for idx, row in val_data.iterrows():
        writer.writerow([row["text"], row["summary"]])
