import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from rdkit import RDLogger
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import traceback 
import random 

from .chem_utils import (
    get_metrics, generate_samples, post_filter_smiles, smiles_to_tensor, MultimodalDataset
)
from create_data import (
    read_raw_data_RNA_Molecule,
    trans, extract_rna_features
)

RDLogger.DisableLog('rdApp.*')

_TRAIN_FEATURES_CACHE = {}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_metrics_to_csv(
        metrics: Dict[str, Any],
        save_path: str,
        stage: str,
        epoch: int,
        fold: int,
        dataset: str,
        split: str,
        extra_info: Optional[Dict[str, Any]] = None,
        run_id: str = None 
):
    ensure_dir(save_path)
    
    if run_id:
        file_name = f"{dataset}_fold{fold}_{run_id}.csv"
    else:
        file_name = f"{dataset}_fold{fold}.csv"
        
    full_path = os.path.join(save_path, file_name)

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "stage": stage,
        "fold": fold,
        "epoch": epoch,
        "split": split,
        **metrics
    }

    if extra_info:
        data.update(extra_info)

    file_exists = os.path.isfile(full_path)
    with open(full_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def load_single_fold_data(args, fold):
    from torch_geometric.loader import DataLoader
    from create_data import (
        read_raw_data_RNA_Molecule,
        trans, extract_rna_features, load_fold_train_stats
    )
    import torch

    dataset = args['dataset']
    batch_size = args['batch_size']
    seed = args['seed']
    val_size = args['val_size']

    split_mode = args.get('split_mode', 'random')
    if split_mode != 'random':
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}; only 'random' is supported."
        )

    rna_emb_dict = extract_rna_features(dataset, force_recalc=False)
    all_folds = read_raw_data_RNA_Molecule(
        dataset, args['n_splits'], seed, val_size
    )

    df_tra, df_val, df_tes = all_folds[fold - 1]

    fold_stats = load_fold_train_stats(
        dataset, fold, df_train=df_tra, split_mode=split_mode
    )
    tra_ds = trans(dataset, df_tra, 'tra', seed, fold, train_stats=fold_stats,
                   rna_embeddings=rna_emb_dict, split_mode=split_mode,
                   max_rna_len=args.get('max_rna_len', 401),
                   n_splits=args['n_splits'], val_size=val_size)
    val_ds = trans(dataset, df_val, 'val', seed, fold, train_stats=fold_stats,
                   rna_embeddings=rna_emb_dict, split_mode=split_mode,
                   max_rna_len=args.get('max_rna_len', 401),
                   n_splits=args['n_splits'], val_size=val_size)
    tes_ds = trans(dataset, df_tes, 'tes', seed, fold, train_stats=fold_stats,
                   rna_embeddings=rna_emb_dict, split_mode=split_mode,
                   max_rna_len=args.get('max_rna_len', 401),
                   n_splits=args['n_splits'], val_size=val_size)

    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id):
        import random
        import numpy as np
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader_tra = DataLoader(tra_ds, batch_size=batch_size, shuffle=True, worker_init_fn=_worker_init, generator=g)
    loader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    loader_tes = DataLoader(tes_ds, batch_size=batch_size, shuffle=False)

    return loader_tra, loader_val, loader_tes

def evaluate_predictor(predictor, loader, device, args, fold):
    predictor.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target = batch.y.view(-1).float()
            
            out = predictor(batch) 
            if isinstance(out, tuple):
                out = out[0]
            out = out.view(-1)
            
            total_loss += loss_fn(out, target).item() * target.size(0)
            all_pred.extend(out.cpu().numpy())
            all_true.extend(target.cpu().numpy())

    metrics = get_metrics(all_true, all_pred)
    return total_loss / max(1, len(loader.dataset)), metrics

def evaluate_generator(generator, predictor, loader, device, args, n_samples_per_rna=5, fold=1):
    default_metrics = {'total_generated': 0, 'valid_ratio': 0.0, 'unique_ratio': 0.0, 'avg_affinity': 0.0}
    if loader is None or (hasattr(loader, 'dataset') and len(loader.dataset) == 0):
        return default_metrics

    try:
        generator.eval()
        predictor.eval()
        
        split_mode = args.get('split_mode', 'random')
        cache_key = f"{args['dataset']}_{split_mode}_fold{fold}"
        if cache_key not in _TRAIN_FEATURES_CACHE:
            train_feat_path = (
                f"data/processed/{args['dataset']}_{split_mode}_fold{fold}_"
                "train_mol_features.npz"
            )
            if os.path.exists(train_feat_path):
                train_data = np.load(train_feat_path, allow_pickle=True)
                _TRAIN_FEATURES_CACHE[cache_key] = {
                    "smiles": set(train_data["smiles"]),
                    "fps": train_data["fingerprints"].tolist(),
                    "scaffolds": set(train_data["scaffolds"])
                }
            else:
                _TRAIN_FEATURES_CACHE[cache_key] = {"smiles": set(), "fps": [], "scaffolds": set()}
        
        cache = _TRAIN_FEATURES_CACHE[cache_key]
        all_smiles, all_affinities = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                

                rna_emb = batch.rna_embedding
                
                initial_smiles_list = batch.smiles_seq
                initial_tokens = smiles_to_tensor(initial_smiles_list, args.get('max_smiles_len', 120)).to(device)

                for _ in range(n_samples_per_rna):
  
                    rna_cond = None
                    if hasattr(predictor, 'encode_rna'):
                        rna_cond = predictor.encode_rna(batch.rna_seq, batch.rna_mask)[0]
                    
                    smiles_list = generate_samples(
                        generator, 
                        torch.randn(batch.num_graphs, args.get('latent_dim', 128), device=device),
                        rna_cond, 
                        initial_smiles_tokens=initial_tokens,
                        max_len=args.get('max_smiles_len', 120)
                    )
                    all_smiles.extend(smiles_list)

                    if hasattr(predictor, 'predict_smiles_list'):
                        
                        target_rna_emb = rna_emb[0] if rna_emb is not None else None
                        
                        preds = predictor.predict_smiles_list(
                            smiles_list, 
                            rna_embedding=target_rna_emb,
                            rna_str=None
                        )
                        all_affinities.extend(preds.cpu().numpy())

        valid_smiles, _ = post_filter_smiles(all_smiles)
        final_affs = np.array(all_affinities) 
        
        strict_novel = [s for s in valid_smiles if s not in cache["smiles"]]
        
        return {
            'total_generated': len(all_smiles),
            'valid_ratio': len(valid_smiles) / len(all_smiles) if all_smiles else 0.0,
            'unique_ratio': len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0,
            'avg_affinity': float(np.mean(final_affs)) if len(final_affs) > 0 else 0.0,
            'strict_novelty': len(strict_novel) / len(valid_smiles) if valid_smiles else 0.0,
            'total_rna': len(loader.dataset)
        }

    except Exception as e:
        print(f"evaluate_generator Error: {e}")
        return default_metrics
