import pandas as pd
from dataclasses import dataclass

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