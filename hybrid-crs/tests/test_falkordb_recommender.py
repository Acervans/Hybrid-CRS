import pytest
import os
import shutil
import pandas as pd

from recsys.falkordb_recommender import FalkorDBRecommender

###########################
# falkordb_recommender.py #
###########################


# Mock test dataset directory
@pytest.fixture(scope="module")
def test_data_dir():
    """Creates a temporary directory with dummy CSV files for testing."""
    dir_path = "./tmp"
    os.makedirs(dir_path, exist_ok=True)

    users_df = pd.DataFrame({"user_id:token": ["1", "2", "3"], "age:int": [25, 30, 35]})
    items_df = pd.DataFrame(
        {
            "item_id:token": ["101", "102", "103", "104"],
            "name:string": [
                "Toy Story",
                "Jumanji",
                "Grumpier Old Men",
                "Waiting to Exhale",
            ],
            "category:string_seq": [
                "Animation Children Comedy",
                "Adventure Children Fantasy",
                "Comedy Romance",
                "Comedy Drama Romance",
            ],
        }
    )
    inters_df = pd.DataFrame(
        {
            "user_id:token": ["1", "1", "2", "2", "3", "3"],
            "item_id:token": ["101", "102", "101", "103", "102", "104"],
            "rating:float": [5.0, 3.0, 4.0, 5.0, 2.0, 4.0],
        }
    )

    users_df.to_csv(f"{dir_path}/test_rec.user", index=False, sep="\t")
    items_df.to_csv(f"{dir_path}/test_rec.item", index=False, sep="\t")
    inters_df.to_csv(f"{dir_path}/test_rec.inter", index=False, sep="\t")

    yield dir_path

    shutil.rmtree(dir_path)


# Mock FalkorDBRecommender graph
@pytest.fixture(scope="module")
def recommender(test_data_dir):
    """Initializes the FalkorDBRecommender with test data and cleans up the graph."""
    rec = FalkorDBRecommender(
        dataset_name="test_rec",
        dataset_dir=test_data_dir,
        graph_name="test-recommender-graph",
        clear=True,
    )
    yield rec

    rec.g.delete()


def test_initialization(recommender):
    """Test if the graph is created and data is ingested correctly."""
    assert recommender.graph_name == "test-recommender-graph"

    user_count = recommender.g.query("MATCH (n:User) RETURN count(n)").result_set[0][0]
    item_count = recommender.g.query("MATCH (n:Item) RETURN count(n)").result_set[0][0]
    rating_count = recommender.g.query(
        "MATCH ()-[r:RATED]->() RETURN count(r)"
    ).result_set[0][0]
    assert user_count == 3
    assert item_count == 4
    assert rating_count == 6
    assert recommender.global_avg is not None


def test_create_user(recommender):
    """Test creating a new user."""
    new_user_id = "4"
    created_id = recommender.create_user(new_user_id)
    assert created_id == new_user_id
    user_exists = recommender.g.query(
        "MATCH (u:User {user_id: $user_id}) RETURN u", params={"user_id": new_user_id}
    ).result_set
    assert len(user_exists) == 1


def test_add_user_properties(recommender):
    """Test adding properties to an existing user."""
    recommender.add_user_properties("1", {"country": "USA"})
    props = recommender.g.query(
        "MATCH (u:User {user_id: '1'}) RETURN u.country"
    ).result_set[0][0]
    assert props == "USA"


def test_add_user_interactions(recommender):
    """Test adding new interactions for a user."""
    user_id = "1"
    initial_interactions = recommender.get_items_by_user(user_id)
    assert "104" not in initial_interactions

    recommender.add_user_interactions(user_id, [("104", 4.5)])

    new_interactions = recommender.get_items_by_user(user_id)
    assert "104" in new_interactions


def test_get_unique_feat_values(recommender):
    """Test retrieving unique values for a feature."""
    # Test for a sequence feature
    categories = recommender.get_unique_feat_values("Item", "category")
    expected_categories = [
        "Animation",
        "Children",
        "Comedy",
        "Adventure",
        "Fantasy",
        "Romance",
        "Drama",
    ]
    assert all(cat in categories for cat in expected_categories)

    # Test for a non-sequence feature
    ages = recommender.get_unique_feat_values("User", "age")
    assert set(ages) == {25, 30, 35}


def test_get_items_by_user(recommender):
    """Test fetching items rated by a specific user."""
    items = recommender.get_items_by_user("2")
    assert set(items) == {"101", "103"}


def test_get_users_by_item(recommender):
    """Test fetching users who rated a specific item."""
    users = recommender.get_users_by_item("101")
    assert set(users) == {"1", "2"}


def test_recommend_contextual(recommender):
    """Test contextual recommendations."""
    recs = recommender.recommend_contextual(
        user_id="3",  # User 3 has not seen item 101 (Comedy)
        item_props={"category": "Comedy"},
        top_n=5,
    )
    assert len(recs) > 0
    # User 3 has seen 102 and 104. Item 101 is a comedy they haven't seen.
    rec_ids = [r[0].properties["item_id"] for r in recs]
    assert "101" in rec_ids


def test_recommend_cf(recommender):
    """Test collaborative filtering recommendations."""
    # User 1 and 2 are similar (both rated item 101).
    # User 2 liked item 103, which user 1 has not seen.
    recs = recommender.recommend_cf(user_id="1", k=1, top_n=5)
    assert len(recs) > 0
    rec_ids = [r[0].properties["item_id"] for r in recs]
    assert "103" in rec_ids


def test_recommend_hybrid(recommender):
    """Test hybrid recommendations."""
    recs = recommender.recommend_hybrid(
        user_id="1", item_props={"category": "Comedy"}, k=1, top_n=5
    )
    assert len(recs) > 0
    # Should contain recommendations from both CF and contextual methods
    rec_ids = {r[0].properties["item_id"] for r in recs}
    assert "103" in rec_ids or "104" in rec_ids


def test_explain_blackbox_recs(recommender):
    """Test explanation generation for a recommendation."""
    # Explain why item '101' might be recommended to user '3'
    # User '3' liked '102' (Adventure Children Fantasy) and '104' (Comedy Drama Romance)
    # Item '101' is (Animation Children Comedy)
    explanations = recommender.explain_blackbox_recs(
        user_id="3", item_id="101", shared_props=["category"], min_rating=0
    )
    assert len(explanations) > 0

    explanation_text = " ".join(map(str, explanations))
    assert "You liked" in explanation_text and "similar category" in explanation_text
