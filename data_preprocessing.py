import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def len_summary(df):
    return len(df["summary"].split())


def len_text(df):
    return len(df["text"].split())


def count_apply(df):
    cnt = 0
    for word in df["summary"].split():
        if word in df["text"]:
            cnt += 1
    return cnt


def count_all_apply(df):
    return len(df["summary"].split())


df_train = pd.read_csv("open/train_data.csv", engine="python", error_bad_lines=False)

# add new columns
nan_list = [0] * len(df_train)
df_train["len_text_split"] = nan_list
df_train["len_summary_split"] = nan_list
df_train["is_it_summary"] = nan_list

# count words of summary and input text
df_train.loc[~df_train["summary"].isna(), ["len_text_split"]] = df_train[
    ~df_train["summary"].isna()
].apply(len_text, axis=1)
df_train.loc[~df_train["summary"].isna(), ["len_summary_split"]] = df_train[
    ~df_train["summary"].isna()
].apply(len_summary, axis=1)

# compare lenth of words between summary and input text
df_train["is_it_summary"] = df_train["len_summary_split"] / df_train["len_text_split"]

# add new columns
df_train["cnt"] = nan_list
df_train["cnt_all"] = nan_list
df_train["is_it_extractive"] = nan_list

# calculate the relativeness between summary and input text
df_train.loc[~df_train["summary"].isna(), ["cnt"]] = df_train[~df_train["summary"].isna()].apply(
    count_apply, axis=1
)
df_train.loc[~df_train["summary"].isna(), ["cnt_all"]] = df_train[
    ~df_train["summary"].isna()
].apply(count_all_apply, axis=1)
df_train["is_it_extractive"] = df_train["cnt"] / df_train["cnt_all"]

# drop improper data
df_train.dropna(inplace=True)
df_train.drop(df_train.loc[df_train["is_it_extractive"] < 0.5].index, inplace=True)
df_train.drop(df_train.loc[df_train["is_it_summary"] > 0.75].index, inplace=True)


train_data, val_data = train_test_split(df_train, test_size=0.1, random_state=42)

# save as train and val data
with open("train_gboot.tsv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["text", "summary"])
    for idx, row in train_data.iterrows():
        writer.writerow([row["text"], row["summary"]])

with open("val_gboot.tsv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["text", "summary"])
    for idx, row in val_data.iterrows():
        writer.writerow([row["text"], row["summary"]])
