import pandas as pd

from src.constants import data_folder


def load_sample_data() -> pd.DataFrame:
    df = pd.read_csv(f"{data_folder}/raw/sample-dataset.csv", header=None)
    return df
