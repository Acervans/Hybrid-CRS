""" Data processing utilities for recommendation datasets.
    Header standardization, data cleaning, normalization and type inference.
"""

# TODO Use Sniffer's dialect to extract necessary info

# TODO USER MUST INDICATE DATA TYPE OF EACH COLUMN, BUT PREVIEW WITH get_datatype

# TODO Let user choose normalization for numericals, autoclean parameters???
# TODO Let user see logs, download final dataset

import os
import re
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
INTER_HEADERS_TEMPLATE = "From these interaction table headers, discern which correspond to user id, item id, and rating (optional). Do NOT exclude suffixes."

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
    rating_column: str | None


class DataType(BaseModel):
    datatype: Literal["token", "token_seq", "float", "float_seq"]


def get_user_headers(headers: List[str]) -> UserHeaders:
    """
    Identify the user ID column from a list of user table headers using LLM inference.

    Args:
        headers (List[str]): List of column headers from the user table

    Returns:
        UserHeaders: Model containing the identified user ID column name
    """
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
    """
    Identify item ID, name, and category columns from a list of item table headers using LLM inference.

    Args:
        headers (List[str]): List of column headers from the item table

    Returns:
        ItemHeaders: Model containing the identified item ID, name, and optional category column names
    """
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
    """
    Identify user ID, item ID, and rating columns from a list of interaction table headers using LLM inference.

    Args:
        headers (List[str]): List of column headers from the interaction table

    Returns:
        InterHeaders: Model containing the identified user ID, item ID, and rating column names
    """
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
    """
    Determine the datatype of a sample list of values using LLM inference.

    Args:
        sample (List[str]): List of sample data values to analyze

    Returns:
        Literal["token", "token_seq", "float", "float_seq"]: The inferred datatype of the sample values
            - token: Single discrete value
            - token_seq: Sequence of discrete values
            - float: Single continuous value
            - float_seq: Sequence of continuous values
    """
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


def normalize(values: np.ndarray, lower: float = 0.0, higher: float = 5.0):
    """
    Normalize an array of values to a specified range.

    Args:
        values (np.ndarray): Array of values to normalize
        lower (float): Lower bound of the target range
        higher (float): Upper bound of the target range

    Returns:
        np.ndarray: Normalized values mapped to the range [lower, higher]
    """
    min_val = values.min()
    max_val = values.max()

    if min_val == max_val:
        return np.full_like(values, higher)

    return (values - min_val) / (max_val - min_val) * (higher - lower) + lower


def process_listlike_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts list-like columns to a space-delimited sequential format.

    Args:
        df (pd.DataFrame): DataFrame to process

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    seq_cols = [col for col in df.columns if col.split(":")[1].endswith("seq")]
    for col in seq_cols:
        non_null = df[col].loc[~df[col].isnull()]
        if not non_null.empty:
            val = non_null.iloc[0]
            if val[0] in ["[("] and val[-1] == "])":
                df[col] = df[col].apply(
                    lambda x: (
                        " ".join(
                            x.replace(" ", "-")
                            for x in re.split(r",\s*", re.sub(r"[\"']", "", x[1:-1]))
                        )
                        if x
                        else x
                    )
                )
    return df


def clean_dataframe(
    dataset: pd.DataFrame,
    except_columns: List[str] = [USER_ID_COL, ITEM_ID_COL],
    logfile: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean a DataFrame using the AutoClean library, excluding specified columns.
    Also processes listlike sequential columns.

    Args:
        dataset (pd.DataFrame): DataFrame to clean
        except_columns (List[str]): List of column names to exclude from cleaning
        logfile (bool): Whether to generate a log file of the cleaning operations
        verbose (bool): Whether to print verbose output during cleaning

    Returns:
        pd.DataFrame: Cleaned DataFrame with preserved columns
    """
    dataset = process_listlike_columns(dataset)
    clean_cols = [col for col in dataset.columns if col not in except_columns]
    if not isinstance(dataset, pd.core.frame.DataFrame):
        dataset = dataset.to_pandas()

    pipeline = AutoClean(
        dataset,
        mode="manual",
        duplicates="auto",
        missing_num=False,
        missing_categ=False,
        encode_categ=False,
        extract_datetime="auto",
        outliers=False,
        logfile=logfile,
        verbose=verbose,
    )
    dataset.loc[:, clean_cols] = pipeline.output.loc[:, clean_cols]

    return pd.from_pandas(dataset)


def sniff_delimiter(sample: str) -> str:
    """
    Sniff the delimiter given a sample string.

    Args:
        sample (str): Sample string to analyze

    Returns:
        str: Detected delimiter
    """
    sniffer = Sniffer()
    dialect = sniffer.sniff(sample)
    return dialect.delimiter


def process_dataset(
    dataset_name: str,
    dataset_dir: str = "./datasets/raw",
    output_dir: str = "./datasets/processed",
    user_headers: UserHeaders = None,
    item_headers: ItemHeaders = None,
    inter_headers: InterHeaders = None,
    normalize_ratings: bool = True,
) -> None:
    """
    Process raw recommendation system datasets by identifying columns, cleaning data, and
    standardizing the format.

    Args:
        dataset_name (str): Name of the dataset to process
        dataset_dir (str): Directory containing raw dataset files.
            Defaults to "./datasets/raw".
        output_dir (str): Directory to store processed dataset files.
            Defaults to "./datasets/processed".
        user_headers (UserHeaders): Pre-identified user table headers.
            If None, they will be automatically identified. Defaults to None.
        item_headers (ItemHeaders): Pre-identified item table headers.
            If None, they will be automatically identified. Defaults to None.
        inter_headers (InterHeaders): Pre-identified interaction table headers.
            If None, they will be automatically identified. Defaults to None.
        normalize_ratings (bool): Whether to normalize rating values to
            the range [0.0, 5.0]. Defaults to True.
    """
    dataset_filename = f"{dataset_dir}/{dataset_name}/{dataset_name}"
    output_folder = f"{output_dir}/{dataset_name}"

    sep = ","
    try:
        # Sniff delimiter from first rows
        with open(f"{dataset_filename}.inter", "r") as f:
            sep = sniff_delimiter(f"{f.readline()}\n{f.readline()}")
    except FileNotFoundError:
        print(f"Mandatory file '{dataset_name}.inter' not found")
        return

    inter_df = pd.read_csv(f"{dataset_filename}.inter", sep=sep)
    if inter_headers is None:
        inter_headers = get_inter_headers(inter_df.columns)

    if inter_headers.rating_column is None:
        inter_df[RATING_COL] = 5.0
        inter_headers.rating_column = RATING_COL
        normalize_ratings = False

    # Standardize interaction headers
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

        # Standardize user headers
        users_df.rename(
            columns={user_headers.user_id_column: USER_ID_COL}, inplace=True
        )
    except FileNotFoundError:
        users_df = None

    try:
        items_df = pd.read_csv(f"{dataset_filename}.item", sep=sep)
        if item_headers is None:
            item_headers = get_item_headers(items_df.columns)

        # Standardize item headers
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

    # Normalize ratings between 0 and 5
    if normalize_ratings and inter_df.loc[:, RATING_COL].max() != 5.0:
        print("Normalizing ratings between 0 and 5...")
        inter_df.loc[:, RATING_COL] = normalize(
            inter_df.loc[:, RATING_COL].values, lower=0.0, higher=5.0
        )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Clean and save processed dataframes to output_dir
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
