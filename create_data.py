import numpy as np
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer, EsmConfig
from utils.chem_utils import (
    smile_to_graph,
    get_smiles_graph,
    compute_fingerprint,
    MultimodalDataset,
    get_scaffold
)
import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import random
import hashlib
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def quantile_discretize(values, n_bins=5):
    return pd.qcut(values, q=n_bins, labels=False, duplicates="drop")

def split_prefix(dataset, split_mode, fold):
    return f"{dataset}_{split_mode}_fold{fold}"


def processed_cache_fingerprint(seed, n_splits, val_size, split_mode, max_rna_len):
    payload = f"seed={int(seed)}|n_splits={int(n_splits)}|val_size={float(val_size):.12g}|"
    payload += f"split_mode={split_mode}|max_rna_len={int(max_rna_len)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def save_split(dataset, fold, df_tra, df_val, df_tes, split_mode="random"):
    os.makedirs("data/processed", exist_ok=True)
    prefix = split_prefix(dataset, split_mode, fold)
    df_tra.to_csv(f"data/processed/{prefix}_tra.csv", index=False)
    df_val.to_csv(f"data/processed/{prefix}_val.csv", index=False)
    df_tes.to_csv(f"data/processed/{prefix}_tes.csv", index=False)


def get_train_y_stats(df_train):
    values = pd.to_numeric(df_train["pKd"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError("Training split contains no valid pKd labels")
    return float(values.min()), float(values.max())


def save_fold_train_stats(dataset, fold, df_train, split_mode="random"):
    stats = get_train_y_stats(df_train)
    stats_path = f"data/processed/{split_prefix(dataset, split_mode, fold)}_train_y_stats.pt"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    torch.save(stats, stats_path)
    return stats


def load_fold_train_stats(dataset, fold, df_train=None, split_mode="random"):
    stats_path = f"data/processed/{split_prefix(dataset, split_mode, fold)}_train_y_stats.pt"
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        return float(stats[0]), float(stats[1])
    if df_train is None:
        train_path = f"data/processed/{split_prefix(dataset, split_mode, fold)}_tra.csv"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing training split for fold {fold}: {train_path}")
        df_train = pd.read_csv(train_path)
    return save_fold_train_stats(dataset, fold, df_train, split_mode=split_mode)

def save_mol_features(dataset, fold, df_train, split_mode="random"):
    save_path = f"data/processed/{split_prefix(dataset, split_mode, fold)}_train_mol_features.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    smiles_list = df_train["SMILES"].dropna().unique().tolist()
    fingerprints = []
    scaffolds = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprints.append(fp.ToBitString())
        scaff = get_scaffold(s)
        if scaff: scaffolds.append(scaff)
    np.savez(save_path, smiles=smiles_list, fingerprints=fingerprints, scaffolds=scaffolds)
    print(f"已保存训练集分子特征至 {save_path}")

def ensure_global_stats(dataset, force_recalc=False):
    stats_dir = os.path.join("data", "processed")
    os.makedirs(stats_dir, exist_ok=True)
    
    stats_path = os.path.join(stats_dir, f"{dataset}_global_stats.pt")
    
    if os.path.exists(stats_path) and not force_recalc:
        print(f"[Info] Loading existing global stats from {stats_path}")
        return torch.load(stats_path)

    print(f"[Info] Global stats not found (or force recalc). Calculating from raw data...")
    
    fpath = f"data/{dataset}/RNA-Molecule.xlsx"
    if not os.path.exists(fpath):
        fpath = f"data/{dataset}/{dataset}.xlsx"
    
    if not os.path.exists(fpath):
        if os.path.exists("RNA-Molecule.xlsx"):
            fpath = "RNA-Molecule.xlsx"
        else:
            raise FileNotFoundError(f"Cannot find raw data excel to compute stats at: {fpath}")

    print(f"[Info] Reading raw data from: {fpath}")
    df_temp = pd.read_excel(fpath)
    
    if "pKd" not in df_temp.columns:
        raise ValueError(f"Column 'pKd' not found in {fpath}")

    g_min = float(df_temp["pKd"].min())
    g_max = float(df_temp["pKd"].max())
    
    print(f"[Global Stats] Dataset: {dataset} | Min: {g_min:.4f} | Max: {g_max:.4f}")
    
    torch.save((g_min, g_max), stats_path)
    print(f"[Info] Saved global stats to {stats_path}")
    
    return g_min, g_max

def extract_rna_features(dataset_name, force_recalc=False):
    save_path = f"data/processed/{dataset_name}_rna_fm_embeddings.npy"
    
    if os.path.exists(save_path) and not force_recalc:
        print(f"[Info] Loading cached RNA embeddings from {save_path}")
        return np.load(save_path, allow_pickle=True).item()

    print("[Info] Extracting RNA features with RNA-FM (Weight Re-mapping Mode)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./pre-train/RNA-FM"
    
    try:
        config = EsmConfig.from_pretrained(model_path)
        tokenizer = EsmTokenizer.from_pretrained(model_path)
        model = EsmModel(config)
        
        weight_file = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(weight_file):
            weight_file = os.path.join(model_path, "model.safetensors")
            
        print(f"[Info] Mapping weights from {weight_file}...")
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("rnafm.", "").replace("layer_norm", "LayerNorm")
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device).eval()
        
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        raise e

    rna_file_path = f"data/{dataset_name}/RNA.xlsx"
    if not os.path.exists(rna_file_path):
        rna_file_path = "data/RNA-Molecule/RNA.xlsx"
    
    try:
        df = pd.read_excel(rna_file_path)
    except ImportError:
        print("\n[Critical] Please install 'openpyxl' to read Excel: pip install openpyxl")
        return None

    rna_dict = {}
    seqs = df["RNA_Sequence"].tolist()
    ids = df["RNA_ID"].tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), 4), desc="RNA-FM Inference"):
            batch_seqs = [str(s).upper() for s in seqs[i : i+4]]
            batch_ids = ids[i : i+4]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state 
            masks = inputs['attention_mask'] 
            
            for j, bid in enumerate(batch_ids):
                real_len = masks[j].sum().item()
                token_embeddings = last_hidden[j, :real_len, :].cpu().numpy().astype(np.float32)
                rna_dict[bid] = token_embeddings

    os.makedirs("data/processed", exist_ok=True)
    np.save(save_path, rna_dict)
    return rna_dict


def read_raw_data_RNA_Molecule(dataset, n_splits, seed, val_size):
    print(f"[Info] 处理 {dataset} 数据（基于样本的交叉验证）")
    fpath = f"data/{dataset}/"
    df_mol = pd.read_excel(fpath + "Molecule.xlsx")
    smiles_map = dict(zip(df_mol["Molecule_ID"], df_mol["SMILES"]))
    df_rnas = pd.read_excel(fpath + "RNA.xlsx")
    df_rnas.set_index("RNA_ID", inplace=True)
    df_all = pd.read_excel(fpath + "RNA-Molecule.xlsx")
    labels = quantile_discretize(df_all["pKd"], n_bins=5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_folds = []
    fold = 1
    for train_idx, test_idx in skf.split(df_all, labels):
        print(f"[Fold {fold}]")
        df_train = df_all.iloc[train_idx].copy()
        df_test = df_all.iloc[test_idx].copy()
        df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=seed, stratify=quantile_discretize(df_train["pKd"]))
        
        df_train = df_train.merge(df_rnas, left_on="RNA_ID", right_index=True)
        df_val = df_val.merge(df_rnas, left_on="RNA_ID", right_index=True)
        df_test = df_test.merge(df_rnas, left_on="RNA_ID", right_index=True)
        
        df_train["SMILES"] = df_train["Molecule_ID"].map(smiles_map)
        df_val["SMILES"] = df_val["Molecule_ID"].map(smiles_map)
        df_test["SMILES"] = df_test["Molecule_ID"].map(smiles_map)
        
        cols = ["RNA_ID", "RNA_Sequence", "Molecule_ID", "SMILES", "pKd"]
        df_train = df_train[cols].dropna()
        df_val = df_val[cols].dropna()
        df_test = df_test[cols].dropna()
        
        save_split(dataset, fold, df_train, df_val, df_test, split_mode="random")
        save_mol_features(dataset, fold, df_train, split_mode="random")
        all_folds.append((df_train, df_val, df_test))
        fold += 1
    return all_folds

def trans(dataset, df_data, tvt_type, seed, fold, train_stats=None, rna_embeddings=None,
          force_reprocess=False, split_mode="random", max_rna_len=401,
          n_splits=10, val_size=0.1):
    drug_ids = df_data["Molecule_ID"].tolist()
    drug_smiles = df_data["SMILES"].tolist()
    rna_ids = df_data["RNA_ID"].tolist()
    rna_seqs = df_data["RNA_Sequence"].tolist()
    labels = df_data["pKd"].values

    smile_graph = get_smiles_graph(dataset)
    drug_fp = [compute_fingerprint(sm) for sm in drug_smiles]

    cache_fp = processed_cache_fingerprint(
        seed, n_splits, val_size, split_mode, max_rna_len
    )
    pyg_data = MultimodalDataset(
        root="data",
        dataset=f"{split_prefix(dataset, split_mode, fold)}_{tvt_type}_L{int(max_rna_len)}_{cache_fp}",
        xd=drug_smiles,
        xt=rna_seqs,
        y=labels,
        drug_fp=drug_fp,
        drug_id=drug_ids,
        rna_id=rna_ids,
        smile_graph=smile_graph,
        y_stats=train_stats,
        rna_embeddings=rna_embeddings,
        force_reprocess=force_reprocess,
        max_rna_len=max_rna_len
    )
    return pyg_data

def prepare_all_data(dataset, n_splits, seed, val_size, split_mode, max_rna_len=401):
    print(f"\n[Data Preparation] Dataset: {dataset} | Mode: {split_mode} | Folds: {n_splits} | Seed: {seed}")
    
    if split_mode != "random":
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}; only 'random' is supported."
        )

    rna_emb_dict = extract_rna_features(dataset, force_recalc=False)

    all_folds = read_raw_data_RNA_Molecule(dataset, n_splits, seed, val_size)

    for fold, (df_tra, df_val, df_tes) in enumerate(all_folds, start=1):       
        fold_stats = save_fold_train_stats(
            dataset, fold, df_tra, split_mode=split_mode
        )
        trans(dataset, df_tra, "tra", seed, fold, train_stats=fold_stats,
              rna_embeddings=rna_emb_dict, force_reprocess=True,
              split_mode=split_mode, max_rna_len=max_rna_len,
              n_splits=n_splits, val_size=val_size)
        trans(dataset, df_val, "val", seed, fold, train_stats=fold_stats,
              rna_embeddings=rna_emb_dict, force_reprocess=True,
              split_mode=split_mode, max_rna_len=max_rna_len,
              n_splits=n_splits, val_size=val_size)
        trans(dataset, df_tes, "tes", seed, fold, train_stats=fold_stats,
              rna_embeddings=rna_emb_dict, force_reprocess=True,
              split_mode=split_mode, max_rna_len=max_rna_len,
              n_splits=n_splits, val_size=val_size)
    print("[Data Preparation] Finished successfully.")

if __name__ == "__main__":
    prepare_all_data(
        dataset="RNA-Molecule",
        n_splits=10,
        seed=42,
        val_size=0.1,
        split_mode="random",
        max_rna_len=401
    )
