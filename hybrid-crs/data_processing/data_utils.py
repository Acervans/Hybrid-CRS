# TODO User Sniffer's dialect to extract necessary info

# TODO USER MUST INDICATE DATA TYPE OF EACH COLUMN, BUT PREVIEW WITH get_datatype

# TODO Let user choose normalization for numericals, autoclean parameters???
# TODO Let user see logs, download final dataset

import os
import numpy as np
import fireducks.pandas as pd

from csv import Sniffer
from AutoClean import AutoClean
from ollama import chat
from pydantic import BaseModel
from typing import List, Literal

MODEL = "qwen2.5:3b"

DATATYPE_TEMPLATE = (
    "From this list of sample data, give the datatype of said data. "
    "Your options are: "
    "'token' (SINGLE discrete value)"
    "'token_seq' (SEQUENCE of discrete values)"
    "'float' (SINGLE continuous value)"
    "'float_seq' (SEQUENCE of continuous values)."
)
USER_HEADERS_TEMPLATE = "From these user table headers, discern which correspond to user id. Do NOT exclude suffixes."
ITEM_HEADERS_TEMPLATE = "From these item table headers, discern which correspond to item id, name and category (optional). Do NOT exclude suffixes."
INTER_HEADERS_TEMPLATE = "From these interaction table headers, discern which correspond to user id, item id, and rating. Do NOT exclude suffixes."

USER_ID_COL = "user_id:token"
ITEM_ID_COL = "item_id:token"
ITEM_NAME_COL = "name:token_seq"
ITEM_CATEGORY_COL = "category"
RATING_COL = "rating:float"

SEP = "\t"


class UserHeaders(BaseModel):
    user_id_column: str


class ItemHeaders(BaseModel):
    item_id_column: str
    name_column: str
    category_column: str | None


class InterHeaders(BaseModel):
    user_id_column: str
    item_id_column: str
    rating_column: str


class DataType(BaseModel):
    datatype: Literal["token", "token_seq", "float", "float_seq"]


def get_user_headers(headers: List[str]) -> UserHeaders:
    response = chat(
        messages=[
            {
                "role": "user",
                "content": (f"{USER_HEADERS_TEMPLATE} Headers: {headers}"),
            }
        ],
        model="qwen2.5:3b",
        format=UserHeaders.model_json_schema(),
        options={"temperature": 0},
    )
    return UserHeaders.model_validate_json(response.message.content)


def get_item_headers(headers: List[str]) -> ItemHeaders:
    response = chat(
        messages=[
            {
                "role": "user",
                "content": (f"{ITEM_HEADERS_TEMPLATE} Headers: {headers}"),
            }
        ],
        model="qwen2.5:3b",
        format=ItemHeaders.model_json_schema(),
        options={"temperature": 0},
    )
    return ItemHeaders.model_validate_json(response.message.content)


def get_inter_headers(headers: List[str]) -> InterHeaders:
    response = chat(
        messages=[
            {
                "role": "user",
                "content": (f"{INTER_HEADERS_TEMPLATE} Headers: {headers}"),
            }
        ],
        model="qwen2.5:3b",
        format=InterHeaders.model_json_schema(),
        options={"temperature": 0},
    )
    return InterHeaders.model_validate_json(response.message.content)


def get_datatype(
    sample: List[str],
) -> Literal["token", "token_seq", "float", "float_seq"]:
    response = chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"{DATATYPE_TEMPLATE} Sample: \n{'\n'.join((f"{x}" for x in sample))}"
                ),
            }
        ],
        model="qwen2.5:3b",
        format=DataType.model_json_schema(),
        options={"temperature": 0},
    )
    return DataType.model_validate_json(response.message.content).datatype


def normalize(values: np.ndarray, lower: float = 1.0, higher: float = 5.0):
    min_val = values.min()
    max_val = values.max()

    return (values - min_val) / (max_val - min_val) * (higher - lower) + lower


def clean_dataframe(
    dataset: pd.DataFrame,
    except_columns: List[str] = [USER_ID_COL, ITEM_ID_COL],
    logfile: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    clean_cols = [col for col in dataset.columns if col not in except_columns]
    if not isinstance(dataset, pd.core.frame.DataFrame):
        dataset = dataset.to_pandas()

    pipeline = AutoClean(
        dataset,
        mode="manual",
        duplicates="auto",
        missing_num="auto",
        missing_categ="auto",
        encode_categ="auto",
        extract_datetime="auto",
        outliers=False,
        logfile=logfile,
        verbose=verbose,
    )
    dataset.loc[:, clean_cols] = pipeline.output.loc[:, clean_cols]

    return pd.from_pandas(dataset)


def process_dataset(
    dataset_name: str,
    dataset_dir: str = "./datasets/raw",
    output_dir: str = "./datasets/processed",
    user_headers: UserHeaders = None,
    item_headers: ItemHeaders = None,
    inter_headers: InterHeaders = None,
    normalize_ratings: bool = False,
) -> None:
    dataset_filename = f"{dataset_dir}/{dataset_name}/{dataset_name}"
    output_folder = f"{output_dir}/{dataset_name}"

    sep = ","
    try:
        # Sniff delimiter from first rows
        with open(f"{dataset_filename}.inter", "r") as f:
            sniffer = Sniffer()
            dialect = sniffer.sniff(f"{f.readline()}\n{f.readline()}")
            sep = dialect.delimiter
    except FileNotFoundError:
        print(f"Mandatory file '{dataset_name}.inter' not found")
        return

    inter_df = pd.read_csv(f"{dataset_filename}.inter", sep=sep)
    if inter_headers is None:
        inter_headers = get_inter_headers(inter_df.columns)

    inter_df.rename(
        columns={
            inter_headers.user_id_column: USER_ID_COL,
            inter_headers.item_id_column: ITEM_ID_COL,
            inter_headers.rating_column: RATING_COL,
        },
        inplace=True,
    )

    try:
        users_df = pd.read_csv(f"{dataset_filename}.user", sep=sep)
        if user_headers is None:
            user_headers = get_user_headers(users_df.columns)

        users_df.rename(
            columns={user_headers.user_id_column: USER_ID_COL}, inplace=True
        )
    except FileNotFoundError:
        users_df = None

    try:
        items_df = pd.read_csv(f"{dataset_filename}.item", sep=sep)
        if item_headers is None:
            item_headers = get_item_headers(items_df.columns)

        rename_cols = {
            item_headers.item_id_column: ITEM_ID_COL,
            item_headers.name_column: ITEM_NAME_COL,
        }
        if item_headers.category_column is not None:
            is_seq = item_headers.category_column.endswith("token_seq")
            rename_cols[item_headers.category_column] = (
                f"{ITEM_CATEGORY_COL}:{"token_seq" if is_seq else "token"}"
            )
        items_df.rename(
            columns=rename_cols,
            inplace=True,
        )
    except FileNotFoundError:
        items_df = None

    if normalize_ratings:
        inter_df.loc[:, RATING_COL] = normalize(
            inter_df.loc[:, RATING_COL].values, lower=1.0, higher=5.0
        )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Clean and persist processed dataframes
    print("Cleaning INTER DataFrame...")
    inter_df = clean_dataframe(inter_df)
    inter_df.to_csv(f"{output_folder}/{dataset_name}.inter", sep=SEP, index=False)
    if users_df is not None:
        print("\nCleaning USER DataFrame...")
        users_df = clean_dataframe(users_df)
        users_df.to_csv(f"{output_folder}/{dataset_name}.user", sep=SEP, index=False)
    if items_df is not None:
        print("\nCleaning ITEM DataFrame...")
        items_df = clean_dataframe(items_df)
        items_df.to_csv(f"{output_folder}/{dataset_name}.item", sep=SEP, index=False)


if __name__ == "__main__":
    item_headers = get_item_headers(
        ["iid:token", "movie_title:token_seq", "genre:token", "release_year:float"]
    )
    assert item_headers.item_id_column == "iid:token"
    assert item_headers.name_column == "movie_title:token_seq"
    assert item_headers.category_column == "genre:token"

    assert (
        get_item_headers(["iid:token", "movie_title:token_seq"]).category_column == None
    )

    user_headers = get_user_headers(
        ["USER_IDENTIFICATION:token", "USER_AGE:float", "USER_NAME:token"]
    )
    assert user_headers.user_id_column == "USER_IDENTIFICATION:token"

    inter_headers = get_inter_headers(
        ["user_id:token", "item_id:token", "rating_value:float", "timestamp:float"]
    )
    assert inter_headers.user_id_column == "user_id:token"
    assert inter_headers.item_id_column == "item_id:token"
    assert inter_headers.rating_column == "rating_value:float"

    assert get_datatype(["My name is Caesar", "The birth of a mother"]) in (
        "token",
        "token_seq",
    )
    assert (
        get_datatype(["Action Comedy", "Adventure", "Drama Animation"]) == "token_seq"
    )
    assert get_datatype(["Comedy", "Adventure", "Animation"]) == "token"
    assert get_datatype([1.12, 2.1, 3.1]) == "float"
    assert get_datatype(["1 2 3", "2 3 3", "3 1 1"]) in ("float_seq", "token_seq")
