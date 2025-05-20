""" Recommendation using FalkorDB for a tabular dataset. Assumes data already preprocessed with `data_processing`
"""

import fireducks.pandas as pd
import numpy as np

from falkordb import FalkorDB
from typing import List, Dict, Any, Tuple, Optional, Literal
from redis.exceptions import ResponseError
from csv import Sniffer
from tqdm import trange


TIMEOUT = 5 * 1000 * 60  # 5 minutes
MIN_R = 10  # Min rating count to be considered reliable


class FalkorDBRecommender:
    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        clear: bool = True,
    ):
        """
        Connect to FalkorDB, select (or create) graph, and ingest the dataset.

        Args:
            dataset_name (str): Name of the dataset. Prefix of CSV file names and graph name in FalkorDB
            dataset_dir (str): Folder containing directory with {dataset_name}.user, .item, .inter CSVs
            host,port,username,password: FalkorDB connection params
            clear (bool): Whether to clear existing data from FalkorDB
        """
        self.dataset_name = dataset_name

        # Connect to FalkorDB and select graph
        self.db = FalkorDB(host=host, port=port, username=username, password=password)
        self.g = self.db.select_graph(dataset_name)

        try:
            empty = (
                clear
                or self.g.ro_query("MATCH (n) RETURN count(*)").result_set[0][0] == 0
            )
            # Clear any existing data
            if clear:
                self.g.delete()
                self.g = self.db.select_graph(dataset_name)
        except ResponseError:
            empty = True

        sep = "\t"
        try:
            # Sniff delimiter from first rows
            with open(f"{dataset_dir}/{dataset_name}/{dataset_name}.inter", "r") as f:
                sniffer = Sniffer()
                dialect = sniffer.sniff(f"{f.readline()}\n{f.readline()}")
                sep = dialect.delimiter
        except FileNotFoundError:
            print(f"Mandatory file '{dataset_name}.inter' not found")

        # Ingest nodes & edges from CSV if empty
        if empty:
            self.inter_df = pd.read_csv(
                f"{dataset_dir}/{dataset_name}/{dataset_name}.inter", sep=sep
            )
            try:
                self.users_df = pd.read_csv(
                    f"{dataset_dir}/{dataset_name}/{dataset_name}.user", sep=sep
                )
            except FileNotFoundError:
                self.users_df = pd.DataFrame(
                    {"user_id:token": self.inter_df["user_id:token"].unique()}
                )
            try:
                self.items_df = pd.read_csv(
                    f"{dataset_dir}/{dataset_name}/{dataset_name}.item", sep=sep
                )
            except FileNotFoundError:
                self.items_df = pd.DataFrame(
                    {"item_id:token": self.inter_df["item_id:token"].unique()}
                )

            # Process feature types, convert sequentials to lists
            self.inter_feats = self._process_columns(self.inter_df)
            self.users_feats = self._process_columns(self.users_df)
            self.item_feats = self._process_columns(self.items_df)

            self._ingest()
            print(
                f"Created graph '{dataset_name}' with {len(self.users_df)} users, {len(self.items_df)} items and {len(self.inter_df)} interactions"
            )
        else:
            self.inter_feats = self._process_columns(
                pd.read_csv(
                    f"{dataset_dir}/{dataset_name}/{dataset_name}.inter",
                    nrows=0,
                    sep=sep,
                )
            )
            try:
                self.user_feats = self._process_columns(
                    pd.read_csv(
                        f"{dataset_dir}/{dataset_name}/{dataset_name}.user",
                        nrows=0,
                        sep=sep,
                    )
                )
            except FileNotFoundError:
                self.user_feats = {}
            try:
                self.item_feats = self._process_columns(
                    pd.read_csv(
                        f"{dataset_dir}/{dataset_name}/{dataset_name}.item",
                        nrows=0,
                        sep=sep,
                    )
                )
            except FileNotFoundError:
                self.item_feats = {}

        # Global average rating
        self.global_avg: float = self.g.query(
            "MATCH ()-[r:RATED]->() RETURN avg(r.rating) AS glb_avg"
        ).result_set[0][0]

    def _process_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Renames columns of a DataFrame without the datatype suffixes,
        returns a dict mapping each feature to its datatype.

        Args:
            df (pd.DataFrame): DataFrame to be processed

        Returns:
            Dict[str, str]: Mapping of feature to datatype
        """
        feats = {x[0]: x[1] for col in df.columns if (x := col.split(":"))}
        df.columns = [col.split(":")[0] for col in df.columns]
        return feats

    def _ingest(self, batch_size: int = 250_000) -> None:
        """
        Ingests users, items and ratings with their respective features as properties.
        Users and items are treated as entities, whereas ratings as RATED relationships.

        Args:
            batch_size (int): Batch size for ingestion of ratings
        """
        # Create indices for fast lookup
        self.g.query("CREATE INDEX ON :User(user_id)")
        self.g.query("CREATE INDEX ON :Item(item_id)")

        # Ingest users
        users = self.users_df.to_dict("records")
        for i in trange(0, len(users), batch_size, desc="Ingesting users"):
            self.g.query(
                "UNWIND $users AS u "
                "MERGE (usr:User {user_id: u.user_id}) "
                "SET usr += u",
                params={"users": users[i : i + batch_size]},
                timeout=TIMEOUT,
            )

        # Ingest items
        items = self.items_df.to_dict("records")
        for i in trange(0, len(items), batch_size, desc="Ingesting items"):
            self.g.query(
                "UNWIND $items AS it "
                "MERGE (itm:Item {item_id: it.item_id}) "
                "SET itm += it",
                params={"items": items[i : i + batch_size]},
                timeout=TIMEOUT,
            )

        # Ingest interactions as RATED edges
        inters = self.inter_df.to_dict("records")
        for i in trange(0, len(inters), batch_size, desc="Ingesting interactions"):
            self.g.query(
                "UNWIND $inters AS r "
                "MATCH (usr:User {user_id: r.user_id}), (itm:Item {item_id: r.item_id}) "
                "MERGE (usr)-[e:RATED]->(itm) "
                "SET e += r {.*, user_id: NULL, item_id: NULL}",
                params={"inters": inters[i : i + batch_size]},
                timeout=TIMEOUT,
            )

        # Precompute PageRank for all Item nodes and store as a property
        print(f"Executing PageRank over '{self.dataset_name}'...")
        self.g.query(
            "CALL algo.pageRank(NULL, 'RATED') YIELD node, score "
            "WHERE node.item_id IS NOT NULL "
            "SET node.pagerank = score",
            timeout=TIMEOUT,
        )

        # Refresh schema for algorithms
        self.g.schema.refresh(1)

    def get_unique_feat_values(
        self, label: Literal["User", "Item", "RATED"], feat: str
    ) -> List[Any]:
        """
        Gets the unique values of a feature for a given label.

        Args:
            label (Literal["User", "Item", "RATED"]): Entity/relationship that has feature `feat`
            feat (str): Feature to get the unique values from

        Returns:
            List[Any]: List of unique values of `feat` for `label`
        """
        is_relationship = False
        if label == "User":
            feats_dict = self.user_feats
        elif label == "Item":
            feats_dict = self.item_feats
        else:
            feats_dict = self.inter_feats
            is_relationship = True

        match_str = f"()-[n:{label}]->()" if is_relationship else f"(n:{label})"
        try:
            q = (
                (
                    f"MATCH {match_str} "
                    f"WITH split(n.{feat}, ' ') AS tokens "
                    "UNWIND tokens AS token "
                    "RETURN collect(DISTINCT token)"
                )
                if feats_dict[feat].endswith("seq")
                else (f"MATCH {match_str} RETURN collect(DISTINCT n.{feat})")
            )
        except KeyError:
            print(f"Feature '{feat}' not a property of '{label}'")
            return []
        return self.g.query(q, timeout=TIMEOUT).result_set[0][0]

    def get_items_by_user(self, user_id: Any) -> List[Any]:
        """
        Gets the list of item IDs a user has interacted with.

        Args:
            user_id (Any): ID of user to query

        Returns:
            List[Any]: List of item IDs
        """
        q = (
            f"MATCH (u:User {{user_id: {user_id}}})-[r:RATED]->(i:Item) "
            f"RETURN i.item_id AS item_id"
        )
        res = self.g.ro_query(q).result_set
        return [row[0] for row in res]

    def get_users_by_item(self, item_id: Any) -> List[Any]:
        """
        Gets the list of user IDs who interacted with an item.

        Args:
            item_id (Any): ID of item to query

        Returns:
            List[Any]: List of user IDs
        """
        q = (
            f"MATCH (u:User)-[r:RATED]->(i:Item {{item_id: {item_id}}}) "
            f"RETURN u.user_id AS user_id"
        )
        res = self.g.ro_query(q).result_set
        return [row[0] for row in res]

    def recommend_contextual(
        self,
        user_id: Any,
        item_props: Dict[str, Any] = {},
        context_weight: float = 0.5,
        score_weight: float = 0.35,
        pagerank_weight: float = 0.15,
        top_n: int = 10,
    ) -> List[Tuple[Any, float]]:
        """
        Contextual recommendations using soft feature overlap + weighted rating score and PageRank.

        Args:
            user_id (Any): ID of target user
            item_props (Dict[str, Any]): e.g. {'category': 'Books'}
            context_weight (float): Weight for number of matching features
            score_weight (float): Weight for weighted score (rating + popularity)
            pagerank_weight (float): Weight for PageRank score
            top_n (int): Number of items to recommend

        Returns:
            List[Tuple[Any, float]]: Ranked pairs of item, score
        """
        # OR clause + CASE condition checking
        conditions = []
        for k, v in item_props.items():
            if self.item_feats.get(k, "").endswith("seq"):
                op = "CONTAINS"
            else:
                op = "="
            if isinstance(v, list):
                conditions.extend((f"itm.{k} {op} '{x}'" for x in v))
            else:
                conditions.append(f"itm.{k} {op} '{v}'")
        or_expr = f"({" OR ".join(conditions)})" if conditions else "true"
        score_expr = (
            " + ".join(
                f"CASE WHEN {condition} THEN 1 ELSE 0 END" for condition in conditions
            )
            if conditions
            else 0
        )

        # Exclude seen items
        seen = self.get_items_by_user(user_id)
        # Item context score, weighted rating score, PageRank
        match_q = (
            f"MATCH (itm:Item) "
            f"WHERE {or_expr} AND NOT itm.item_id IN {seen} "
            f"OPTIONAL MATCH ()-[r:RATED]->(itm) "
            f"WITH itm, avg(r.rating) AS avgRating, count(r) AS cnt "
            f"RETURN itm, ({score_expr}) AS contextScore, "
            f"((cnt * avgRating) + ({MIN_R} * {self.global_avg})) / (cnt + {MIN_R}) AS weightedScore, "  # Bayesian average
            f"itm.pagerank AS pageRank"
        )
        res = self.g.ro_query(match_q, timeout=TIMEOUT).result_set
        if not res:
            return []

        # Merge scores, fill nan
        score_arr = np.array(
            [
                [
                    row[1],  # context_score
                    row[2],  # weighted_score
                    row[3],  # pagerank
                ]
                for row in res
            ],
            dtype=np.float32,
        )
        score_arr[:, 1][np.isnan(score_arr[:, 1])] = 0.0

        # Min max normalization
        mins = score_arr.min(axis=0)
        maxs = score_arr.max(axis=0)
        score_arr = (score_arr - mins) / (maxs - mins + 1e-9)

        # Weighted scores
        weights = np.array(
            [context_weight, score_weight, pagerank_weight],
            dtype=np.float32,
        )
        score_arr = score_arr @ weights

        # Map to item_id
        scores = [(res[i][0], score_arr[i]) for i in range(len(res))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def recommend_cf(
        self, user_id: Any, min_rating: float = 3.0, k: int = 10, top_n: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        Collaborative filtering using FalkorDB graph queries.

        Args:
            user_id (Any): ID of target user
            min_rating (float): Minimum rating threshold to consider items as liked
            k (int): Number of nearest neighbors to consider
            top_n (int): Number of items to recommend

        Returns:
            List[Tuple[Any, float]]: Ranked pairs of item, score
        """
        # Find top-k similar users based on co-rated items
        q = (
            f"MATCH (u1:User {{user_id: {user_id}}})-[r1:RATED]->(i:Item)<-[r2:RATED]-(u2:User) "
            f"WHERE r1.rating >= {min_rating} AND r2.rating >= {min_rating} AND u1 <> u2 "
            f"WITH u2, count(i) AS sharedItems "
            f"ORDER BY sharedItems DESC "
            f"LIMIT {k} "
            f"RETURN u2.user_id AS neighborId"
        )
        neighbor_res = self.g.ro_query(
            q,
            timeout=TIMEOUT,
        ).result_set
        if not neighbor_res:
            return []

        neighbor_ids = [row[0] for row in neighbor_res]

        # Exclude seen items
        seen = self.get_items_by_user(user_id)
        # Get recommendations from similar neighbors, sort by weighted rating
        q_recs = (
            f"MATCH (n:User)-[r:RATED]->(i:Item) "
            f"WHERE n.user_id IN {neighbor_ids} AND NOT i.item_id IN {seen} AND r.rating >= {min_rating} "
            f"WITH i, avg(r.rating) as avgRating, count(r.rating) as cnt "
            f"WITH i, ((cnt * avgRating) + ({k} * {self.global_avg})) / (cnt + {k}) AS weightedScore "  # Bayesian average
            f"RETURN i, weightedScore "
            f"ORDER BY weightedScore DESC "
            f"LIMIT {top_n}"
        )
        recs = self.g.ro_query(
            q_recs,
            timeout=TIMEOUT,
        ).result_set

        return [(row[0], row[1] / 5.0) for row in recs if row[0] is not None]

    def recommend_hybrid(
        self,
        user_id: Any,
        item_props: Dict[str, Any] = {},
        min_rating: float = 3.0,
        k: int = 5,
        top_n: int = 10,
    ) -> List[Tuple[Any, float]]:
        """
        Hybrid recommendations: contextual + CF recommendations
        blend by interleaving ranks, prioritizing intersections.

        Args:
            user_id (Any): ID of target user
            item_props (Dict[str, Any]): e.g. {'category': 'Books'} for contextual recommendations
            min_rating (float): Minimum rating threshold to consider items as liked
            k (int): Number of nearest neighbors to consider for CF recommendations
            top_n (int): Number of items to recommend

        Returns:
            List[Tuple[Any, float]]: Ranked pairs of item, score
        """
        ctx = self.recommend_contextual(user_id, item_props, top_n=top_n)
        cf = self.recommend_cf(user_id, k=k, min_rating=min_rating, top_n=top_n)
        ctx_ids = set(x[0].properties["item_id"] for x in ctx)
        cf_ids = set(x[0].properties["item_id"] for x in cf)

        hybrid = []
        for i, score in enumerate(np.linspace(1, 0, max(len(ctx), len(cf)))):
            if i < len(ctx):
                hybrid.append(
                    (
                        ctx[i][0],
                        1.0 if ctx[i][0].properties["item_id"] in cf_ids else score,
                    )
                )
            if i < len(cf):
                hybrid.append(
                    (
                        cf[i][0],
                        1.0 if cf[i][0].properties["item_id"] in ctx_ids else score,
                    )
                )
        return hybrid[:top_n]

    def explain_blackbox_recs(
        self,
        user_id: Any,
        item_id: Any,
        shared_props: Optional[List[str]] = None,
        min_rating: float = 3.0,
        rating_threshold: float = 3.0,
        top_feat_exp: int = 5,
        top_collab_exp: int = 5,
    ) -> List[str]:
        """
        Explain a recommendation from any black-box recommender using graph-based reasoning only.

        Args:
            user_id (Any): ID of the user for whom the item was recommended
            item_id (Any): The recommended item
            shared_props (Optional[List[str]]): Features to check for mutual properties
            min_rating (float): Only consider RATED edges with rating >= this threshold
            rating_threshold (float): Minimum rating for general acceptance explanation
            top_feat_exp (int): Number of explanations through feature similarity
            top_collab_exp (int): Number of explanations through collaborative path evidence

        Returns:
            List[str]: List of natural language explanation strings
        """
        explanations = []
        similar_items = {}

        # 1. Feature similarity via shared properties (e.g. category, brand, etc.)
        if shared_props:
            rec_props = (
                self.g.query(f"MATCH (rec:Item {{item_id: {item_id}}}) RETURN rec")
                .result_set[0][0]
                .properties
            )
            for prop in shared_props:
                if self.item_feats.get(prop, "").endswith("seq"):
                    prop_values = rec_props.get(prop, "").split(" ")
                    condition = " OR ".join(
                        f"i.{prop} CONTAINS '{x}'" for x in prop_values
                    )
                else:
                    condition = f"i.{prop} = '{rec_props.get(prop, "")}'"
                query_feat = (
                    f"MATCH (u:User {{user_id: {user_id}}})-[r:RATED]->(i:Item) "
                    f"WHERE r.rating >= {min_rating} AND {condition} "
                    f"RETURN i.{prop} AS value, i "
                    f"ORDER BY r.rating DESC "
                    f"LIMIT {top_feat_exp}"
                )
                feat_res = self.g.ro_query(
                    query_feat,
                    timeout=TIMEOUT,
                ).result_set
                for value, liked_item in feat_res:
                    explanations.append(
                        f"You liked '{liked_item.properties["name"]}' where {prop} = {value}, and this item has a similar {prop} = {rec_props[prop]}."
                    )
                    liked_id = liked_item.properties["item_id"]
                    if liked_id in similar_items:
                        similar_items[liked_id].append((prop, value))
                    else:
                        similar_items[liked_id] = [(prop, value)]

        # 2. Collaborative multi-hop path evidence, also check if item shares prop
        match_expr = (
            f"(i.item_id IN {list(similar_items.keys())})" if similar_items else "false"
        )
        query_cf = (
            f"MATCH (u:User {{user_id: {user_id}}})-[r1:RATED]->(i:Item)<-[r2:RATED]-(u2:User)-[r3:RATED]->(rec:Item {{item_id: {item_id}}}) "
            f"WHERE r1.rating >= {min_rating} AND r2.rating >= {min_rating} AND r3.rating >= {min_rating} "
            f"WITH count(DISTINCT u2) AS numUsers, i AS item, {match_expr} AS sharesProp "
            f"RETURN numUsers, item, sharesProp "
            f"ORDER BY sharesProp DESC, numUsers DESC "
            f"LIMIT {top_collab_exp}"
        )
        cf_res = self.g.ro_query(
            query_cf,
            timeout=TIMEOUT,
        ).result_set
        if cf_res:
            for num_users, item, shares_prop in cf_res:
                similar_str = (
                    f"It also has similar features: {(
                        ", ".join(f"{x[0]} = {x[1]}" for x in similar_items[item.properties["item_id"]])
                    )}"
                    if shares_prop
                    else ""
                )
                explanations.append(
                    (
                        f"{num_users} other users who also enjoyed '{item.properties["name"]}' "
                        f"also liked this item. {similar_str}"
                    )
                )

        # 3. Popularity of item as general acceptance
        query_pop = (
            f"MATCH (:User)-[r:RATED]->(rec:Item {{item_id: {item_id}}}) "
            f"RETURN avg(r.rating) AS avgRating, count(r) AS totalRatings"
        )
        pop_result = self.g.ro_query(query_pop, timeout=TIMEOUT).result_set
        if pop_result:
            avg_rating, total = pop_result[0]
            if avg_rating and avg_rating >= rating_threshold:
                explanations.append(
                    f"This item has an average rating of {round(avg_rating, 2)}/5.0 based on {total} user ratings."
                )

        return (
            explanations
            if explanations
            else ["No strong explanation found based on current graph data."]
        )


if __name__ == "__main__":
    dataset = "ml-10m"
    datasets_folder = "../data_processing/datasets/processed"
    frec = FalkorDBRecommender(dataset, datasets_folder, clear=False)

    res = frec.get_unique_feat_values("Item", "category")
    print(res)

    res = frec.recommend_contextual(
        user_id=71559,
        item_props={"category": ["Animation", "Action"]},
        top_n=10,
        context_weight=0.5,
        score_weight=0.35,
        pagerank_weight=0.15,
    )
    for node, score in res:
        print(node, score)

    res = frec.recommend_cf(
        user_id=71559,
        top_n=10,
        k=10,
    )
    for node, score in res:
        print(node, score)

    res = frec.recommend_hybrid(
        user_id=71559,
        item_props={"category": "Animation"},
        top_n=10,
        k=10,
    )
    for node, score in res:
        print(node, score)

    for node, score in res:
        props = node.properties
        ctx_props = props.copy()
        del ctx_props["item_id"], ctx_props["pagerank"], ctx_props["name"]

        exps = frec.explain_blackbox_recs(
            user_id=71559,
            item_id=props["item_id"],
            shared_props=ctx_props.keys(),
            min_rating=3.0,
            rating_threshold=3.0,
        )
        print(f"Explanations for {props["name"]}")
        for exp in exps:
            print(f"\t{exp}")
