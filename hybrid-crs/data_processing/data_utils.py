""" Data processing utilities for recommendation datasets.
    Header standardization, data cleaning, normalization and type inference.
"""

import os
import re
import numpy as np
import fireducks.pandas as pd

from csv import Sniffer, QUOTE_MINIMAL
from AutoClean import AutoClean
from ollama import chat
from pydantic import BaseModel
from typing import List, Literal, Iterable, Optional

MODEL = "qwen2.5:3b"

DATATYPE_TEMPLATE = """
From this column of sample data, determine the datatype. Your options are: 
'token': SINGLE discrete value (ID, genre...)
'token_seq': SEQUENCE of discrete values (description, name, list of genres...)
'float': SINGLE numerical value (price, timestamp...)
'float_seq': SEQUENCE of numerical values (vector, array...).
"""

USER_HEADERS_TEMPLATE = "From these user table headers, discern which correspond to user id. Do NOT exclude suffixes."
ITEM_HEADERS_TEMPLATE = "From these item table headers, discern which correspond to item id, name and category (optional). Do NOT exclude suffixes."
INTER_HEADERS_TEMPLATE = "From these interaction table headers, discern which correspond to user id, item id, and rating (optional). Do NOT exclude suffixes."

USER_ID = "user_id:token"
ITEM_ID = "item_id:token"
ITEM_NAME = "name:token_seq"
ITEM_CATEGORY = "category"
RATING = "rating:float"

SEP = "\t"


class UserHeaders(BaseModel):
    user_id_column: str


class ItemHeaders(BaseModel):
    item_id_column: str
    name_column: str
    category_column: Optional[str] = None


class InterHeaders(BaseModel):
    user_id_column: str
    item_id_column: str
    rating_column: Optional[str] = None


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
                    f"{DATATYPE_TEMPLATE}\nSample: \n{'\n'.join((f"{x}" for x in sample))}"
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
            if val[0] in "[(" and val[-1] in "])":
                df[col] = df[col].apply(
                    lambda x: (
                        " ".join(
                            x.strip().replace(" ", "-")
                            for x in re.split(r",\s*", re.sub(r"[\"']", "", x[1:-1]))
                        )
                        if x
                        else x
                    )
                )
    return df


def clean_dataframe(
    dataset: pd.DataFrame,
    except_columns: Iterable[str] = (USER_ID, ITEM_ID),
    logfile: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean a DataFrame using the AutoClean library, excluding specified columns.
    Also processes listlike sequential columns.

    Args:
        dataset (pd.DataFrame): DataFrame to clean
        except_columns (Iterable[str]): Column names to exclude from cleaning
        logfile (bool): Whether to generate a log file of the cleaning operations
        verbose (bool): Whether to print verbose output during cleaning

    Returns:
        pd.DataFrame: Cleaned DataFrame with preserved columns
    """
    dataset = process_listlike_columns(dataset)
    clean_cols = [col for col in dataset.columns if col not in set(except_columns)]
    if not isinstance(dataset, pd.core.frame.DataFrame):
        dataset = dataset.to_pandas()

    pipeline = AutoClean(
        dataset,
        mode="manual",
        duplicates="auto",
        missing_num=False,
        missing_categ="most_frequent",
        encode_categ=False,
        extract_datetime="auto",
        outliers=False,
        logfile=logfile,
        verbose=verbose,
    )
    dataset.loc[:, clean_cols] = pipeline.output.loc[:, clean_cols]

    return pd.from_pandas(dataset)


def sniff_delimiter(sample: str | list[str]) -> str:
    """
    Sniff the delimiter given a sample string or list of strings.

    Args:
        sample (str | list[str]): Sample string or list of strings to analyze

    Returns:
        str: Detected delimiter
    """
    sniffer = Sniffer()
    if isinstance(sample, str):
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except:
            return ","
    elif isinstance(sample, list):
        count = dict()
        for string in sample:
            try:
                dialect = sniffer.sniff(string)
                count[dialect.delimiter] = count.get(dialect.delimiter, 0) + 1
            except:
                continue
        return max(count.items(), key=lambda x: x[1])[0] if count else ","


def process_dataset(
    dataset_name: str,
    dataset_dir: str = "./datasets/raw/example",
    output_dir: str = "./datasets/processed/example",
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
            Defaults to "./datasets/raw/example".
        output_dir (str): Directory to store processed dataset files.
            Defaults to "./datasets/processed/example".
        user_headers (UserHeaders): Pre-identified user table headers.
            If None, they will be automatically identified. Defaults to None.
        item_headers (ItemHeaders): Pre-identified item table headers.
            If None, they will be automatically identified. Defaults to None.
        inter_headers (InterHeaders): Pre-identified interaction table headers.
            If None, they will be automatically identified. Defaults to None.
        normalize_ratings (bool): Whether to normalize rating values to
            the range [0.0, 5.0]. Defaults to True.
    """

    def replace_datatype(left_colname: str, right_colname: str, default: str = "token"):
        dtype = right_colname.split(":")[1] if ":" in right_colname else default
        return f"{left_colname.split(":")[0]}:{dtype}"

    dataset_filename = os.path.normpath(f"{dataset_dir}/{dataset_name}")
    inter_infer = user_infer = item_infer = False

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
        inter_infer = True

    if inter_headers.rating_column is None:
        inter_df[RATING] = 5.0
        inter_headers.rating_column = RATING
        normalize_ratings = False

    # Standardize interaction headers
    rating_col = (
        replace_datatype(RATING, inter_headers.rating_column, "float")
        if not inter_infer
        else RATING
    )
    inter_df.rename(
        columns={
            inter_headers.user_id_column: (
                replace_datatype(USER_ID, inter_headers.user_id_column, "token")
                if not inter_infer
                else USER_ID
            ),
            inter_headers.item_id_column: (
                replace_datatype(ITEM_ID, inter_headers.item_id_column, "token")
                if not inter_infer
                else ITEM_ID
            ),
            inter_headers.rating_column: rating_col,
        },
        inplace=True,
    )

    try:
        users_df = pd.read_csv(f"{dataset_filename}.user", sep=sep)
        if user_headers is None:
            user_headers = get_user_headers(users_df.columns)
            user_infer = True

        # Standardize user headers
        users_df.rename(
            columns={
                user_headers.user_id_column: (
                    replace_datatype(USER_ID, user_headers.user_id_column, "token")
                    if not user_infer
                    else USER_ID
                )
            },
            inplace=True,
        )
    except FileNotFoundError:
        users_df = None

    try:
        items_df = pd.read_csv(f"{dataset_filename}.item", sep=sep)
        if item_headers is None:
            item_headers = get_item_headers(items_df.columns)
            item_infer = True

        # Standardize item headers
        rename_cols = {
            item_headers.item_id_column: (
                replace_datatype(ITEM_ID, item_headers.item_id_column, "token")
                if not item_infer
                else ITEM_ID
            ),
            item_headers.name_column: (
                replace_datatype(ITEM_NAME, item_headers.name_column, "token_seq")
                if not item_infer
                else ITEM_NAME
            ),
        }
        if item_headers.category_column is not None:
            rename_cols[item_headers.category_column] = replace_datatype(
                ITEM_CATEGORY, item_headers.category_column, "token"
            )
        items_df.rename(
            columns=rename_cols,
            inplace=True,
        )
    except FileNotFoundError:
        items_df = None

    # Normalize ratings between 0 and 5
    if normalize_ratings and inter_df.loc[:, rating_col].max() != 5.0:
        print("Normalizing ratings between 0 and 5...")
        inter_df.loc[:, rating_col] = normalize(
            inter_df.loc[:, rating_col].values, lower=0.0, higher=5.0
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Clean and save processed dataframes to output_dir
    print("Cleaning INTER DataFrame...")
    inter_df = clean_dataframe(inter_df)
    inter_df.to_csv(
        f"{output_dir}/{dataset_name}.inter",
        sep=SEP,
        index=False,
        quoting=QUOTE_MINIMAL,
        escapechar="\\",
    )
    if users_df is not None:
        print("\nCleaning USER DataFrame...")
        users_df = clean_dataframe(users_df)
        users_df.to_csv(
            f"{output_dir}/{dataset_name}.user",
            sep=SEP,
            index=False,
            quoting=QUOTE_MINIMAL,
            escapechar="\\",
        )
    if items_df is not None:
        print("\nCleaning ITEM DataFrame...")
        items_df = clean_dataframe(items_df)
        items_df.to_csv(
            f"{output_dir}/{dataset_name}.item",
            sep=SEP,
            index=False,
            quoting=QUOTE_MINIMAL,
            escapechar="\\",
        )
