import os
import sys
import torch
import torch.distributed as dist
import numpy as np
import warnings
import pathlib
import importlib
import itertools

from collections import OrderedDict
from torch import load, cuda, device
from logging import getLogger

from recbole.config import Config
from recbole.data.dataset import Interaction, Dataset
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

warnings.filterwarnings("ignore", category=FutureWarning)
# use CPU if CUDA unavailable
load_device = device("cpu") if not cuda.is_available() else None


def parse_model(model: str):
    try:
        model_class = get_model(model)
    except ValueError as v:
        # Import from current directory
        if importlib.util.find_spec(model):
            module = importlib.import_module(model)
            model_class = getattr(module, model)
        else:
            raise v

    return model_class


def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    model = parse_model(model)

    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    try:
        model = (get_model(config["model"]) if isinstance(model, str) else model)(
            config, train_data._dataset
        ).to(config["device"])
    except ValueError:
        print(f"Model {model} not found")

    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process


def load_data_and_model(
    load_model,
    preload_dataset=None,
    update_config=None,
    use_training=False,
    verbose=False,
):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        load_model (dict | str): Preloaded checkpoint or path to saved model.
        preload_dataset (Dataset): Preloaded dataset.
        update_config (dict): Config entries to update.
        use_training (bool): Whether to use training set or full dataset.
        verbose (bool): Whether to log data preparation.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    sys.path.insert(1, str(pathlib.Path(__file__).parent.resolve()))

    checkpoint = load_model
    if isinstance(load_model, str):
        checkpoint = load(load_model, map_location=load_device)

    config: Config = checkpoint["config"]
    if update_config:
        for key, value in update_config.items():
            config[key] = value

    if config["data_path"]:
        config["data_path"] = config["data_path"].replace("\\", "/")

    if not cuda.is_available():
        config["device"] = "cpu"

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    if verbose:
        logger = getLogger()
        logger.info(config)

    dataset = preload_dataset or create_dataset(config)

    if verbose:
        logger.info(dataset)

    init_seed(config["seed"], config["reproducibility"])

    model = parse_model(config["model"])

    train_data = valid_data = test_data = None
    if use_training:
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = model(config, train_data._dataset).to(config["device"])
    else:
        model = model(config, dataset).to(config["device"])

    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data


def evaluate_saved_model(saved_model, update_config=None, evaluation_mode="full"):
    load_model = load(saved_model, map_location=load_device)
    eval_args = load_model["config"]["eval_args"]

    if eval_args:
        eval_args["mode"] = evaluation_mode
    else:
        config["eval_args"] = {"mode": evaluation_mode}

    config, model, _, _, _, test_data = load_data_and_model(
        load_model, update_config=update_config, use_training=True, verbose=True
    )

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model evaluation
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
        model_file=saved_model,
    )

    getLogger().info(set_color("test result", "yellow") + f": {test_result}")


def full_sort_scores(
    model: torch.nn.Module,
    device: torch.device,
    dataset: Dataset,
    uid_inter: dict,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predicts scores of all tracks for a given user

    Args:
        model (torch.nn.Module): Recommendation model
        device (torch.device): Device where the model is loaded
        uid_inter (dict): Internal user ID mapping
        batch_size (int): Number of items to process at once

    Returns:
        np.ndarray: Predicted track scores
    """
    item_feats = dataset.get_item_feature()
    scores = list()
    for i in range(0, dataset.item_num, batch_size):
        interaction = Interaction(uid_inter)
        item_feat = item_feats[i : i + batch_size]
        interaction = interaction.repeat_interleave(len(item_feat))
        interaction.update(item_feat.repeat(1))
        scores.append(model.predict(interaction.to(device)).detach().cpu().numpy())
    return np.concatenate(scores)


def scores_to_recommendations(
    scores: torch.Tensor | np.ndarray, dataset: Dataset, cutoff: int
) -> list[tuple]:
    """Assigns respective tracks to the generated scores

    Args:
        scores (torch.Tensor | np.ndarray): Tracks' scores
        cutoff (int): Number of tracks to recommend

    Returns:
        list[tuple]: Recommendations (score, track_id)
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy().flatten()
    scores = scores[1:]

    item_ids = dataset.id2token(dataset.iid_field, list(range(1, dataset.item_num)))

    sorted_idx = np.argsort(scores)[: -(cutoff + 1) : -1]
    return list(zip(scores[sorted_idx], item_ids[sorted_idx]))


def get_recommendations(
    user_id: int,
    model: str,
    device: torch.device,
    dataset: Dataset,
    predict_args: dict = dict(),
    cutoff: int = 10,
) -> list[tuple]:
    """Obtains recommendations for a given user using the selected model

    Args:
        user_id (int): ID of user to get recommendations for
        model (torch.nn.Module): Recommendation model
        device (torch.device): Device where the model is loaded
        predict_args (dict): Additional recommendation arguments
        cutoff (int): Number of tracks to recommend

    Returns:
        list[tuple]: Recommendations (score, track_id)
    """
    uid_series = torch.Tensor(dataset.token2id(dataset.uid_field, [str(user_id)]))

    model.eval()
    uid_inter = {dataset.uid_field: uid_series}
    try:
        scores = model.full_sort_predict(uid_inter, **predict_args)
    except NotImplementedError:
        scores = full_sort_scores(model, device, uid_inter)

    recs = scores_to_recommendations(scores, dataset, cutoff)

    return recs


def prepare_dataset(
    model: str,
    config_file: str,
    config_dict: dict,
    dataset_name: str | None = None
):
    """Prepares dataset and data splits once for reuse."""
    if dataset_name:
        config_dict['dataset'] = dataset_name
    config = Config(model=model, config_file_list=[config_file], config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return dataset, train_data, valid_data, test_data


def build_trainer(
    model: str,
    config_file: str,
    base_config_dict: dict,
    overrides: dict,
    train_data
):
    """Build trainer with given config overrides."""
    config_dict = {**base_config_dict, **overrides}
    config = Config(model=model, config_file_list=[config_file], config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    model_cls = get_model(model)
    model_inst = model_cls(config, train_data.dataset).to(config['device'])

    trainer_cls = get_trainer(config['MODEL_TYPE'], model)
    trainer = trainer_cls(config, model_inst)

    return config, trainer


def retrain_on_dataset(
    model: str,
    config_file: str,
    best_params: dict,
    dataset_name: str | None = None,
    valid_set: Dataset | None = None,
    save_best_model_path: str | None = None
):
    """Retrain model on full dataset and save if path is given."""
    config_dict = {
        'model': model,
        **best_params,
        'saved': True,
        'load_best_model': True,
        'eval_args': {
            'split': {'RS': [1.0, 0.0, 0.0]},
            'order': 'RO',
            'group_by': 'user',
            'mode': 'full'
        }
    }
    if save_best_model_path:
        config_dict['checkpoint_dir'] = os.path.dirname(save_best_model_path)

    config = Config(model=model, dataset=dataset_name, config_file_list=[config_file], config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)

    model_cls = get_model(model)
    model_inst = model_cls(config, train_data.dataset).to(config['device'])

    trainer_cls = get_trainer(config['MODEL_TYPE'], model)
    trainer = trainer_cls(config, model_inst)

    if save_best_model_path:
        os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)
        trainer.saved_model_file = save_best_model_path

    # Fit on full dataset, validate on valid set used for hyperparam searching
    _, scores = trainer.fit(train_data, valid_set, saved=True)

    print(f"Final model saved at: {save_best_model_path}")

    return scores


def hyperparam_exhaustive_search(
    model: str,
    config_file: str,
    param_grid: dict,
    dataset_name: str | None = None,
    save_best_model_path: str | None = None,
) -> tuple[dict, OrderedDict]:
    """
    Manually performs exhaustive grid search for RecBole models.

    Args:
        model (str): Model name (e.g., "EASE")
        config_file (str): Path to RecBole YAML config file
        param_grid (dict): Dict of hyperparameter choices
        dataset_name (str | None): Dataset name in place of config dataset
        save_best_model_path (str | None): Path to save the best model (if any)

    Returns:
        tuple[dict, OrderedDict]: Best hyperparameters and best test scores
    """
    base_config_dict = {'model': model}
    _dataset, train_data, valid_data, _ = prepare_dataset(
        model, config_file, base_config_dict, dataset_name
    )

    best_score = None
    best_params = None

    for values in itertools.product(*param_grid.values()):
        param_comb = dict(zip(param_grid.keys(), values))
        print(f"Testing params: {param_comb}")

        _config, trainer = build_trainer(
            model=model,
            config_file=config_file,
            base_config_dict=base_config_dict,
            overrides={
                **param_comb,
                'saved': False,
                'load_best_model': False
            },
            train_data=train_data
        )

        val_score, scores = trainer.fit(train_data, valid_data, saved=False)

        if best_score is None or val_score > best_score:
            best_score = val_score
            best_params = param_comb

        print(f"Validation score: {val_score}")
        print(f"Scores: {scores}")
        print("-" * 40)

    print(f"Best params found: {best_params}")
    print("Retraining on full dataset...")

    test_result = retrain_on_dataset(
        model=model,
        dataset_name=dataset_name,
        valid_set=valid_data,
        config_file=config_file,
        best_params=best_params,
        save_best_model_path=save_best_model_path
    )

    print("Test result:", test_result)
    return best_params, test_result


if __name__ == "__main__":
    # run_recbole(
    #     "EASE", dataset="ml-100k", config_file_list=["config/generic.yaml"], saved=False
    # )

    best_params, test_scores = hyperparam_exhaustive_search(
        model="EASE",
        config_file="config/generic.yaml",
        param_grid={"reg_weight": [1.0, 10.0, 100.0, 250.0, 500.0, 1000.0]},
        dataset_name="ml-100k",
        save_best_model_path="saved/ease-best.pth",
    )
    print("Best params:", best_params)
    print("Test scores:", test_scores)

    # hyperopt_tuning("EASE", ['config/generic.yaml'], 'ease.hyper', algo="exhaustive")

    # config, model, dataset, _train_set, _valid_set, _test_set = load_data_and_model(
    #     load_model='./saved/EASE-May-22-2025_04-34-38.pth',
    #     preload_dataset=None,
    # )

    # res = get_recommendations(
    #     1,
    #     model,
    #     load_device,
    #     dataset,
    # )
    # print(res)

    # # Load another model with same dataset
    # config, model, _, _, _, _ = load_data_and_model(
    #     load_model='./saved/ItemKNN-May-22-2025_05-46-48.pth',
    #     preload_dataset=dataset,
    # )
    # res = get_recommendations(
    #     1,
    #     model,
    #     load_device,
    #     dataset,
    # )
    # print(res)
