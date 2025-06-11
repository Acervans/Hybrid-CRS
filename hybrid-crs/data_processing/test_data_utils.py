from data_utils import (
    get_item_headers,
    get_user_headers,
    get_inter_headers,
    get_datatype,
)


def test_get_item_headers():
    item_headers = get_item_headers(
        ["iid:token", "movie_title:token_seq", "genre:token", "release_year:float"]
    )
    assert item_headers.item_id_column == "iid:token"
    assert item_headers.name_column == "movie_title:token_seq"
    assert item_headers.category_column == "genre:token"

    assert (
        get_item_headers(["iid:token", "movie_title:token_seq"]).category_column == None
    )


def test_get_user_headers():
    user_headers = get_user_headers(
        ["USER_IDENTIFICATION:token", "USER_AGE:float", "USER_NAME:token"]
    )
    assert user_headers.user_id_column == "USER_IDENTIFICATION:token"


def test_get_inter_headers():
    inter_headers = get_inter_headers(
        ["user_id:token", "item_id:token", "rating_value:float", "timestamp:float"]
    )
    assert inter_headers.user_id_column == "user_id:token"
    assert inter_headers.item_id_column == "item_id:token"
    assert inter_headers.rating_column == "rating_value:float"


def test_get_datatype():
    assert get_datatype(["My name is Caesar", "The birth of a mother"]) in (
        "token",
        "token_seq",
    )
    assert (
        get_datatype(["Action Comedy", "Adventure", "Drama Animation"]) == "token_seq"
    )
    assert get_datatype(["Comedy", "Adventure", "Animation"]) == "token"
    assert get_datatype([1.12, 2.1, 3.1]) == "float"
    assert get_datatype([1, 2, 3]) in ("token", "float")
    assert get_datatype(["1 2 3", "2 3 3", "3 1 1"]) in ("float_seq", "token_seq")
    assert get_datatype(["[1 2 3]", "[2, 3, 3]", "(3 1 1)"]) in (
        "float_seq",
        "token_seq",
    )
