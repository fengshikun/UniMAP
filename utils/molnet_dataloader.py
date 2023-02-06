import os
from typing import List
import numpy as np

import pandas as pd
from utils.sampl_datasets import load_sampl_iupac
from deepchem.molnet import  load_clearance, \
    load_qm7, load_qm8, load_qm9, load_muv
    
# load_delaney, load_hiv, load_lipo, load_bbbp, load_tox21
from utils.clintox_datasets import load_clintox
from utils.sider_datasets import load_sider

from utils.delaney_datasets import load_delaney # esol
from utils.hiv_datasets import load_hiv
from utils.lipo_datasets import load_lipo
from utils.bbbp_datasets import load_bbbp

from utils.tox21_datasets import load_tox21
from utils.toxcast_datasets import load_toxcast
from utils.bace_datasets import load_bace_classification, load_bace_regression
from utils.malaria_datasets import load_malaria
from utils.cep_datasets import load_cep


from rdkit import Chem

MOLNET_DIRECTORY = {
    "bace_classification": {
        "dataset_type": "classification",
        "load_fn": load_bace_classification,
        "split": "scaffold",
    },
    "bace_regression": {
        "dataset_type": "regression",
        "load_fn": load_bace_regression,
        "split": "scaffold",
    },
    "bbbp": {
        "dataset_type": "classification",
        "load_fn": load_bbbp,
        "split": "scaffold",
    },
    "clearance": {
        "dataset_type": "regression",
        "load_fn": load_clearance,
        "split": "scaffold",
    },
    "clintox": {
        "dataset_type": "classification",
        "load_fn": load_clintox,
        "split": "scaffold",
        # "tasks_wanted": ["CT_TOX"],
    },
    "delaney": {
        "dataset_type": "regression",
        "load_fn": load_delaney,
        "split": "scaffold",
    },
    "hiv": {
        "dataset_type": "classification",
        "load_fn": load_hiv,
        "split": "scaffold",
    },
    # pcba is very large and breaks the dataloader
    #     "pcba": {
    #         "dataset_type": "classification",
    #         "load_fn": load_pcba,
    #         "split": "scaffold",
    #     },
    "lipo": {
        "dataset_type": "regression",
        "load_fn": load_lipo,
        "split": "scaffold",
    },
    "qm7": {
        "dataset_type": "regression",
        "load_fn": load_qm7,
        "split": "random",
    },
    "qm8": {
        "dataset_type": "regression",
        "load_fn": load_qm8,
        "split": "random",
    },
    "qm9": {
        "dataset_type": "regression",
        "load_fn": load_qm9,
        "split": "random",
    },
    "sider": {
        "dataset_type": "classification",
        "load_fn": load_sider,
        "split": "scaffold",
    },
    "tox21": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "scaffold",
        # "tasks_wanted": ["SR-p53"],
    },
    "toxcast": {
        "dataset_type": "classification",
        "load_fn": load_toxcast,
        "split": "scaffold",
        # "tasks_wanted": ["SR-p53"],
    },
    "muv": {
        "dataset_type": "classification",
        "load_fn": load_muv,
        "split": "scaffold",
        # "tasks_wanted": ["SR-p53"],
    },
    "sampl": {
        "dataset_type": "regression",
        "load_fn": load_sampl_iupac,
        "split": "scaffold",
    },
    "malaria": {
        "dataset_type": "regression",
        "load_fn": load_malaria,
        "split": "scaffold",
    },
    "cep": {
        "dataset_type": "regression",
        "load_fn": load_cep,
        "split": "scaffold",
    }
    
}


def get_dataset_info(name: str):
    return MOLNET_DIRECTORY[name]


def load_molnet_dataset(
    name: str,
    split: str = None,
    tasks_wanted: List = None,
    df_format: str = "chemberta",
    seed: int = 0,
):
    """Loads a MolNet dataset into a DataFrame ready for either chemberta or chemprop.

    Args:
        name: Name of MolNet dataset (e.g., "bbbp", "tox21").
        split: Split name. Defaults to the split specified in MOLNET_DIRECTORY.
        tasks_wanted: List of tasks from dataset. Defaults to `tasks_wanted` in MOLNET_DIRECTORY, if specified, or else all available tasks.
        df_format: `chemberta` or `chemprop`

    Returns:
        tasks_wanted, (train_df, valid_df, test_df), transformers

    """
    load_fn = MOLNET_DIRECTORY[name]["load_fn"]
    tasks, splits, transformers = load_fn(
        featurizer="Raw", splitter=split or MOLNET_DIRECTORY[name]["split"], seed=seed,
    )

    # Default to all available tasks
    if tasks_wanted is None:
        tasks_wanted = MOLNET_DIRECTORY[name].get("tasks_wanted", tasks)
    # tasks_wanted = ['Nervous system disorders']
    print(f"Using tasks {tasks_wanted} from available tasks for {name}: {tasks}")

    # tasks_wanted = tasks.copy()
    # # for sp in splits:
    # sp = splits[2] # test set
    # for ti, task in enumerate(tasks):
    #     all_valid_label = sp.y[:, ti][sp.w[:, ti]!=0]
    #     if (np.unique(all_valid_label)).size <= 1:
    #         tasks_wanted.remove(task)



    return (
        tasks_wanted,
        [
            make_dataframe(
                s,
                MOLNET_DIRECTORY[name]["dataset_type"],
                tasks,
                tasks_wanted,
                df_format,
            )
            for s in splits
        ],
        transformers,
    )


def write_molnet_dataset_for_chemprop(
    name: str, split: str = None, tasks_wanted: List = None, data_dir: str = None
):
    """Writes a MolNet dataset to separate train, val, test CSVs ready for chemprop.

    Args:
        name: Name of MolNet dataset (e.g., "bbbp", "tox21").
        split: Split name. Defaults to the split specified in MOLNET_DIRECTORY.
        tasks_wanted: List of tasks from dataset. Defaults to all available tasks.
        data_dir: Location to write CSV files. Defaults to /tmp/molnet/{name}/.

    Returns:
        tasks_wanted, (train_df, valid_df, test_df), transformers, out_paths

    """
    if data_dir is None:
        data_dir = os.path.join("/tmp/molnet/", name)
    os.makedirs(data_dir, exist_ok=True)

    tasks, dataframes, transformers = load_molnet_dataset(
        name, split=split, tasks_wanted=tasks_wanted, df_format="chemprop"
    )

    out_paths = []
    for split_name, df in zip(["train", "val", "test"], dataframes):
        path = os.path.join(data_dir, f"{split_name}.csv")
        out_paths.append(path)
        df.to_csv(path, index=False)

    return tasks, dataframes, transformers, out_paths



def to_dataframe(data_dict) -> pd.DataFrame:
    """Construct a pandas DataFrame containing the data from this Dataset.

    Returns
    -------
    pd.DataFrame
      Pandas dataframe. If there is only a single feature per datapoint,
      will have column "X" else will have columns "X1,X2,..." for
      features.  If there is only a single label per datapoint, will
      have column "y" else will have columns "y1,y2,..." for labels. If
      there is only a single weight per datapoint will have column "w"
      else will have columns "w1,w2,...". Will have column "ids" for
      identifiers.
    """
    X = data_dict["X"]
    y = data_dict["y"]
    w = data_dict["w"]
    if len(X.shape) == 1 or X.shape[1] == 1:
      columns = ['X']
    else:
      columns = [f'X{i+1}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    if len(y.shape) == 1 or y.shape[1] == 1:
      columns = ['y']
    else:
      columns = [f'y{i+1}' for i in range(y.shape[1])]
    y_df = pd.DataFrame(y, columns=columns)
    if len(w.shape) == 1 or w.shape[1] == 1:
      columns = ['w']
    else:
      columns = [f'w{i+1}' for i in range(w.shape[1])]
    w_df = pd.DataFrame(w, columns=columns)
    
    ids = data_dict["smile_ids"]
    smiles_ids_df = pd.DataFrame(ids, columns=['smile_ids'])
    if 'iupac_ids' in data_dict:
        iupac_ids = data_dict["iupac_ids"]
        iupac_ids_df = pd.DataFrame(iupac_ids, columns=['iupac_ids'])
        return pd.concat([X_df, y_df, w_df, smiles_ids_df, iupac_ids_df], axis=1, sort=False)
    else:
        return pd.concat([X_df, y_df, w_df, smiles_ids_df], axis=1, sort=False)
    
    
    
    


def make_dataframe(
    dataset, dataset_type, tasks, tasks_wanted, df_format: str = "chemberta"
):
    iupac = False
    if len(dataset.ids.shape) == 2 and dataset.ids.shape[1] == 2: # contains iupac
        iupac = True
        data_dict = {}
        data_dict["iupac_ids"] = dataset.ids[:,1]
        data_dict["smile_ids"] = dataset.ids[:,0]
        data_dict["y"] = dataset.y
        data_dict["w"] = dataset.w
        data_dict["X"] = dataset.X
        # no need for X; rdkit Chem rdchem Mol object
        # df = pd.DataFrame(data_dict)
        df = to_dataframe(data_dict)
    elif 'MUV' in tasks[0]: # muv dataset
        iupac = True
        data_dict = {}
        data_dict["smile_ids"] = dataset.ids
        data_dict["y"] = dataset.y
        data_dict["w"] = dataset.w
        data_dict["X"] = dataset.X
        df = to_dataframe(data_dict)
    else:
        df = dataset.to_dataframe()
        
    if len(tasks) == 1:
        mapper = {"y": tasks[0]}
    else:
        
        tasks_wanted_index_dict = {}
        for task in tasks_wanted:
            tasks_wanted_index_dict[task] = tasks.index(task)
        mapper = {f"y{y_i+1}": task for task, y_i in tasks_wanted_index_dict.items()}
    df.rename(mapper, axis="columns", inplace=True)
    

    # Canonicalize SMILES
    # smiles_list = [Chem.MolToSmiles(s, isomericSmiles=True) for s in df["X"]]
    smiles_list = [Chem.MolToSmiles(s) for s in df["X"]]
    # smiles_list = [Chem.MolToSmiles(s, isomericSmiles=False, canonical=True) for s in df["X"]]

    # smiles_list_2 = [Chem.MolToSmiles(s, isomericSmiles=False, canonical=True) for s in df["X"]]
    # df['smile_ids'] = smiles_list_2

    # df['smile_ids'] = smiles_list
    # Convert labels to integer for classification
    labels = df[tasks_wanted]
    if dataset_type == "classification":
        labels = labels.astype(int)

    elif dataset_type == "regression":
        labels = labels.astype(float)

    if iupac:
        return df

    if df_format == "chemberta":
        if len(tasks_wanted) == 1:
            labels = labels.values.flatten()
        else:
            # Convert labels to list for simpletransformers multi-label
            labels = labels.values.tolist()
        return pd.DataFrame({"text": smiles_list, "labels": labels})
    elif df_format == "chemprop":
        df_out = pd.DataFrame({"smiles": smiles_list})
        for task in tasks_wanted:
            df_out[task] = labels[task]
        return df_out
    else:
        raise ValueError(df_format)