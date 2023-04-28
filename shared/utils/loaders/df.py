import gzip
import json

import pandas as pd


def parse(path):
    g = gzip.open(path, "r")
    for line in g:
        yield json.loads(line)


def json_gzip_into_df(path: str):
    parsed_data = parse(path)
    return pd.DataFrame(parsed_data)


def csv_gzip_into_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def df_into_csv_gzip(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_csv(path + ".csv.gz", compression="gzip", index=False, header=True)

        return True

    except Exception:
        return False
