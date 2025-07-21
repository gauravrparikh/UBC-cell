from .torch_extensions import CachedPartLabels, LoaderBundle, TensorToDictDatasetAdapter
import scanpy as sc
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from protopnet.visualization import KeyReturningDict

DEFAULT_DATA_PATH = "/usr/xtmp/jcd97/datasets/mini_data.h5ad" #"/usr/xtmp/UBC2025/data/human_cell_atlas.h5ad"
label_column = 'cell_type'
n_top_genes = 100_000

def load_data(data_path):
    """
    Load the example dataset for experiment 1.
    
    Returns:
        ad.AnnData: The loaded AnnData object containing the dataset.
    """
    try:
        adata = sc.read_h5ad(data_path)
        print(f"Data loaded successfully from {data_path}")
        return adata
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def preprocess_data(adata):
    """
    Preprocess the AnnData object by normalizing and log-transforming the data.
    
    Args:
        adata (ad.AnnData): The AnnData object to preprocess.
        
    Returns:
        ad.AnnData: The preprocessed AnnData object.
    """
    print("Grabbing highly variable")
    sc.experimental.pp.highly_variable_genes(adata, n_top_genes=10_000, chunksize=10) # reason about this. It does mean adjusted variance of some sort. 
    print("Normalizing")
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("Log")
    sc.pp.log1p(adata)
    print("Scale")
    sc.pp.scale(adata, max_value=10)
    print("Returning")
    return adata[:, adata.var['highly_variable']]

def train_dataloaders(
    # data_path: Union[str, pathlib.Path] = os.environ.get("CUB200_DIR", "CUB_200_2011"),
    data_path="/usr/xtmp/lam135/datasets/CUB_200_2011_2/",
    train_dir: str = "train",
    val_dir: str = "validation",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=True,
    color_patch_params={},
    debug: bool = False,
    debug_forbid_dir: str = "debug_folder/forbid",
    debug_remember_dir: str = "debug_folder/remember",
    excluded_classes=[
        "Miscellaneuous",
        "unannoted", 
        "Splatter", 
        "Deep-layer intratelencephalic", 
        "MGE interneuron", 
        "Astrocyte", 
        "Amygdala excitatory"
    ]
):

    print("About to run load")
    all_data = load_data(DEFAULT_DATA_PATH)
    print("About to run preprocess")
    all_data = preprocess_data(all_data)
    print("Finished Pre-processing")
    # np.save("/usr/xtmp/UBC2025/data/human_cell_atlas_processed_X.npy", all_data.X)
    # np.save("/usr/xtmp/UBC2025/data/human_cell_atlas_processed_y.npy", all_data.obs[label_column])
    # print("Saved objects")

    all_data_X = torch.tensor(all_data.X.toarray()).float()
    
    y_names = all_data.obs[label_column]
    kept_samples = ~y_names.isin(excluded_classes)
    print(f"Before exclusion: {y_names.shape}, {np.unique(y_names)}")
    y_names = y_names[kept_samples]
    print(f"After exclusion: {y_names.shape}, {np.unique(y_names)}")
    

    counter = 0
    name_to_int = {}
    for v in y_names:
        if v not in name_to_int:
            name_to_int[v] = counter
            counter += 1

    y_vals = torch.tensor([name_to_int[v] for v in y_names])
    X_kept = all_data_X[kept_samples]

    n_samples = X_kept.shape[0]

    idx = torch.randperm(y_vals.nelement())
    y_vals = y_vals[idx]
    all_data_X = X_kept[idx]
    train_X, val_X = all_data_X[:n_samples // 2], all_data_X[n_samples // 2:]
    train_y, val_y = y_vals[:n_samples // 2], y_vals[n_samples // 2:]

    print(counter + 1, "Classes")

    class_name_ref_dict = {}
    for name in name_to_int.keys():
        class_name_ref_dict[name_to_int[name]] = name
    # for classname in os.listdir(data_path + "train/"):
    #     class_ind, class_name = classname.split(".")
    #     class_ind = int(class_ind) - 1
    #     class_name = " ".join(class_name.split("_"))
    #     class_name_ref_dict[class_ind] = class_name
    
    train_dataset = TensorToDictDatasetAdapter(TensorDataset(train_X, train_y))
    train_dataloader = DataLoader(train_dataset, batch_size=224, shuffle=True)

    val_dataset = TensorToDictDatasetAdapter(TensorDataset(val_X, val_y))
    val_dataloader = DataLoader(val_dataset, batch_size=224, shuffle=True)

    loader_packet = LoaderBundle(
        train_dl=train_dataloader,
        train_loader_no_aug=train_dataloader,
        val_dl=val_dataloader,
        proj_dl=train_dataloader,
        num_classes=counter + 1,
        image_size=(1,1),
        class_name_ref_dict=KeyReturningDict(class_name_ref_dict)
    )


    return loader_packet