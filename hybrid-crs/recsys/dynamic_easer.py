import numpy as np

from tqdm import tqdm

from incremental_fm import IncrementalFM
from cornac.models import (
    VAECF,
    BiVAECF,
    BPR,
    BaselineOnly,
    MostPop,
    GlobalAvg,
    PMF,
    MF,
    ItemKNN,
    UserKNN,
    EASE,
    EFM,
    NeuMF,
)


class DynamicEASEr(Recommender):
    """Dynamic EASEᴿ for incremental training of the EASEᴿ 
    (Embarrasingly Shallow Auto-Encoders in reverse) model.

    Reference
    ----------
    * Jeunen, O. Van Balen, J. & Goethals, B. 
    Embarrassingly shallow auto-encoders for dynamic collaborative filtering. 
    User Modeling and User-Adapted Interaction (2022).
    """

    incremental: bool = True

    def __init__(self, p=None, k=40, l2_reg_w0=2.0, l2_reg_w=8.0, l2_reg_V=16.0, learn_rate=0.004, incremental_fit=False, seed=None):
        self.incremental_fit = incremental_fit
        self.seed = seed

        super().initialize(use_index=True)

        Recommender.__init__(self, "IncrementalFM")


    def fit(self, train_set: cornac.data.Dataset, val_set: cornac.data.Dataset = None):
        """Fit the model to a dataset.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if not self.trainable:
            return self

        # static baseline; w/o updating the model
        if self.static:
            return

        events = []
        for u, i, r in train_set.uir_iter():
            if r > 0:
                event = Event(User(u[0]), Item(i[0]), value=1.0)
                events.append(event)

        self.register_users([User(u) for u in train_set.uid_map.values()])
        self.register_items([Item(i) for i in train_set.iid_map.values()])

        if self.incremental_fit:
            [
                self.update(e) 
                for e in tqdm(events)
            ]
        else:
            self.fit_events(
                events,
                user_feature=bool(train_set.user_feature),
                item_feature=bool(train_set.item_feature)
            )

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            return 1 - FMRecommender.score(self, User(user_idx), list(self.iid_map.values()))
        else:
            return 1 - FMRecommender.score(self, User(user_idx), [self.iid_map[item_idx]])[0]



if __name__ == "__main__":

    # data = movielens.load_feedback(variant="100k")
    data = amazon_digital_music.load_feedback()

    # Split the data into training and testing sets
    eval_method = RatioSplit(data=data, test_size=0.2, rating_threshold=3.0, seed=100)
    # eval_method = CrossValidation(data=data, n_folds=5, rating_threshold=3.0, seed=123)
    # eval_method = StratifiedSplit(
    #     data=data, fmt="UIR", test_size=0.2, rating_threshold=3.0, seed=123)

    models = [
        # IncrementalFM(p=0, k=10, l2_reg_w0=1, l2_reg_w=2,
        #               l2_reg_V=4, learn_rate=0.002, incremental_fit=False),
        # BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
        # DynamicEASEr(),
        EASE(),
        NeuMF(),
        # VAECF(k=10),
        # BiVAECF(k=10),
        # BaselineOnly(),
        MostPop(),
        # GlobalAvg(),
        # PMF(k=10),
        MF(k=10, backend='pytorch', optimizer='adam'),
        UserKNN(k=10),
        ItemKNN(k=10),
    ]

    metrics = [Precision(k=10), Recall(
        k=10), NDCG(k=10), HitRatio(k=10), AUC()]

    cornac.Experiment(
        eval_method=eval_method,
        models=models,
        metrics=metrics,
        user_based=True,
        verbose=True,
        save_dir='./cornac_storage'
    ).run()

    import shutil
    shutil.rmtree("./cornac_storage", ignore_errors=True)

    # import numpy as np

    # # p = user_features+item_features+event_features
    # np.random.seed(1)
    # recommender = FMRecommender(p=9, k=10)
    # recommender.initialize(use_index=True)

    # import time

    # a = time.time()
    # for i in range(1000):
    #     user = User(i, feature=np.array([1,0,1]))
    #     item = Item(i, feature=np.array([2,1,1]))
    #     event = Event(user, item, context=np.array([2,1,1]))
    #     recommender.register(user)
    #     recommender.register(item)
    #     recommender.update(event)
    # print(time.time()-a)

    # print(recommender.recommend(user, np.array([0]), context=np.array([0,4,0])))

    # recs1 = recommender.recommend(user, np.array(np.arange(1000)), context=np.array([0,4,0]))

    # # p = user_features+item_features+event_features
    # recommender = IncrementalFM(p=9, k=10, seed=1)
    # recommender.initialize(use_index=True)

    # a = time.time()

    # users = []
    # items = []
    # events = []
    # for i in range(1000):
    #     users.append(User(i, feature=np.array([1,0,1])))
    #     items.append(Item(i, feature=np.array([2,1,1])))
    #     events.append(Event(user, item, context=np.array([2,1,1])))

    # recommender.register_users(users)
    # recommender.register_items(items)
    # recommender.fit_events(events)
    # print(time.time()-a)

    # # specify target user and list of item candidates

    # print(recommender.recommend(user, np.array([0]), context=np.array([0,4,0])))
    # # => (sorted candidates, scores)

    # recs2 = recommender.recommend(user, np.array(np.arange(1000)), context=np.array([0,4,0]))
    # print(np.array_equal(recs1, recs2))
