import os

import pandas as pd
from dataclasses import dataclass

ALL_RECORDS = []
RANDOM_RECORDS = []

@dataclass
class ImageRecord:
    image: str
    center_x_1: int
    center_y_1: int
    polomer_1: int
    center_x_2: int
    center_y_2: int
    polomer_2: int
    center_x_3: int
    center_y_3: int
    polomer_3: int
    center_x_4: int
    center_y_4: int
    polomer_4: int


def load_random_records(csv_path, n=100):
    global RANDOM_RECORDS

    df = pd.read_csv(csv_path)
    n = min(n, len(df))
    sampled_df = df.sample(n=n)

    records = []

    for _, row in sampled_df.iterrows():
        row_dict = row.to_dict()
        records.append(ImageRecord(**row_dict))

    RANDOM_RECORDS = records
    return records

def load_valid_records(csv_path, data_folder, n=100):
    """
    Načíta záznamy kým nemá n platných (súbor existuje na disku).
    Záznamy sa vyberajú náhodne z celého datasetu.
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1)  # náhodné poradie
    valid = []
    for _, row in df.iterrows():
        record = ImageRecord(**row.to_dict())
        path = os.path.normpath(os.path.join(data_folder, record.image.replace("\\", "/")))
        if os.path.exists(path):
            valid.append(record)
        if len(valid) >= n:
            break
    print(f"Načítaných {len(valid)} platných záznamov z {len(df)} celkovo")
    return valid
