import pandas as pd

from racecast.config import FEATURE_MATRIX_PATH

def get_split():

    df = pd.read_parquet(FEATURE_MATRIX_PATH)



    train = df[df["year"] <= 2021]
    val = df[df["year"] == 2022]
    test = df[df["year"].between(2023, 2025)]
    # df[df["year"] == 2026] — set aside, not part of train/val/test
    return train, val, test
