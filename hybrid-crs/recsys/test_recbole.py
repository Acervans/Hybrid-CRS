import os
import recbole
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm

if __name__ == '__main__':

    models_configs = [
        ('BPR',      'config/generic.yaml'),
        ('EASE',     'config/generic.yaml'),
        ('FM',       'config/generic.yaml'),
        ('LightGCN', 'config/generic.yaml'),
        ('MultiVAE', 'config/generic.yaml'),
        ('xDeepFM',  'config/generic.yaml'),
        ('WideDeep', 'config/generic.yaml'),
        ('Pop',      'config/once.yaml'),
        ('Random',   'config/once.yaml'),
    ]

    models, configs = zip(*models_configs)

    # os.system(f"""python run_recbole_group.py --[model_list]={','.join(models)} --[dataset]=[dataset] --[config_files]=[config_files] --[valid_latex]=[valid_latex] --[test_latex]=[test_latex]""")
    os.system(f"""python run_recbole_group.py --[model_list]={','.join(
        models)} --[dataset]=ml-100k --[config_files]={','.join(configs)}""")
