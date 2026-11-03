import torch
from torch_geometric.data import InMemoryDataset, Data
import os
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold  
import pandas as pd
import random
import numpy as np
from scipy import stats

try:
    from utils import sascorer
except ImportError:
    sascorer = None

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    torch.use_deterministic_algorithms(True, warn_only=False) 
    print(f"[Info] Global Random Seed set to {seed} with Deterministic Algorithms enabled.")

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_feature(atom):
    try:
        symbol = one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                                       'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                                       'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                                       'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
        degree = one_of_k_encoding(atom.GetDegree(), list(range(11)))
        total_hs = one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        try:
            implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        except:
            implicit_valence = [0] * 11
        is_aromatic = [atom.GetIsAromatic()]
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(charge) or np.isinf(charge): charge = 0.0
        except:
            charge = 0.0
        is_h_acceptor = 1 if atom.GetSymbol() in ['N', 'O', 'F'] else 0
        base_features = symbol + degree + total_hs + implicit_valence + is_aromatic
        phys_features = [charge, is_h_acceptor]
        return np.array(base_features + phys_features)
    except Exception as e:
        print(f"Atom Feature Error: {e}")
        return np.zeros(80)

def get_bond_feature(bond):
    bt = bond.GetBondType()
    return (
        [bt == Chem.rdchem.BondType.SINGLE,
         bt == Chem.rdchem.BondType.DOUBLE,
         bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC,
         bond.GetIsConjugated(),
         bond.IsInRing()]
    )

def get_atom_feature_dims():
    return 80 

from rdkit.Chem import Descriptors

def get_physchem_features(mol):
    try:
        return np.array([
            Descriptors.MolLogP(mol),      
            Descriptors.TPSA(mol),         
            Descriptors.NumHDonors(mol),    
            Descriptors.NumHAcceptors(mol), 
            Descriptors.MolWt(mol) / 100.0  
        ], dtype=np.float32)
    except:
        return np.zeros(5, dtype=np.float32)

def compute_fingerprint(smiles, fp_type="morgan", n_bits=2048, include_physchem=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits + (5 if include_physchem else 0), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    fp_arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    if include_physchem:
        phys_features = get_physchem_features(mol)
        return np.concatenate([fp_arr, phys_features]).astype(np.float32)
    return fp_arr

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def molecule_quality_metrics(smiles, reference_smiles=None):
    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    if mol is None:
        return np.nan, np.nan, np.nan

    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = np.nan

    sa = np.nan
    if sascorer is not None:
        try:
            sa = float(sascorer.calculateScore(mol))
        except Exception:
            pass

    similarity = np.nan
    if reference_smiles is not None:
        ref_mol = Chem.MolFromSmiles(str(reference_smiles))
        if ref_mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                similarity = float(DataStructs.TanimotoSimilarity(ref_fp, fp))
            except Exception:
                pass

    return qed, sa, similarity


def is_optimization_success(delta_pkd, final_sa, tanimoto,
                            delta_threshold=0.3, sa_cutoff=5.5,
                            similarity_cutoff=0.2):
    try:
        return bool(
            float(delta_pkd) >= float(delta_threshold)
            and np.isfinite(final_sa)
            and float(final_sa) <= float(sa_cutoff)
            and np.isfinite(tanimoto)
            and float(tanimoto) >= float(similarity_cutoff)
        )
    except (TypeError, ValueError):
        return False

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception: return None

def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            mol = Chem.MolFromSmiles(smile, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
            if mol is None: raise ValueError("Invalid SMILES")
        AllChem.ComputeGasteigerCharges(mol)
        features = []
        target_dim = get_atom_feature_dims()
        for atom in mol.GetAtoms():
            f = get_atom_feature(atom)
            if len(f) < target_dim: f = np.pad(f, (0, target_dim - len(f)), mode='constant')
            elif len(f) > target_dim: f = f[:target_dim]
            features.append(f)
        edges = []
        edge_features = []
        if mol.GetNumBonds() > 0:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                feature = get_bond_feature(bond)
                edges.append([i, j])
                edge_features.append(feature)
                edges.append([j, i])
                edge_features.append(feature)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)
        x = torch.tensor(np.array(features), dtype=torch.float)
        return x, edge_index, edge_attr
    except Exception as e:
        print(f"Failed to process SMILES: {smile} - {e}")
        return None, None, None

def smiles_to_graph_data(smiles):
    x, edge_index, edge_attr = smile_to_graph(smiles)
    if x is None: return None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def get_smiles_graph(dataset):
    fpath = f'data/{dataset}/'
    cache_file = f"data/processed/smile_graph_edge.pt" 
    if os.path.exists(cache_file):
        print(f"[Info] Loading cached smile_graph from {cache_file}")
        return torch.load(cache_file)
    print("[Info] Parsing SMILES and computing Charges...")
    df_lig = pd.read_excel(fpath + "Molecule.xlsx") 
    all_smiles = set(df_lig['SMILES'].values)
    smile_graph = {}
    missing_smiles = []
    for sm in all_smiles:
        x, edge_index, edge_attr = smile_to_graph(sm)
        if x is not None: smile_graph[sm] = (x, edge_index, edge_attr)
        else: missing_smiles.append(sm)
    if missing_smiles: print(f"[Error] Failed SMILES count: {len(missing_smiles)}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save(smile_graph, cache_file)
    print(f"[Info] Saved smile_graph cache to {cache_file}")
    return smile_graph

def valid_mol_filter(smiles: str, min_ring_size: int = 0):
    if not smiles or len(smiles) == 0: return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    try: Chem.SanitizeMol(mol)
    except: return False
    return True

def post_filter_smiles(smiles_list, train_smiles_set=None):
    valid = [s for s in smiles_list if valid_mol_filter(s)]
    unique = list(set(valid))
    n_total = len(smiles_list) + 1e-9
    n_valid = len(valid)
    n_unique = len(unique)
    metrics = {'valid_ratio': n_valid / n_total, 'unique_ratio': n_unique / (n_valid + 1e-9), 'novelty': 0.0}
    if train_smiles_set:
        novel = [s for s in unique if s not in train_smiles_set]
        metrics['novelty'] = len(novel) / (n_unique + 1e-9)
    return valid, metrics

def generate_samples(*args, **kwargs): return []
def generate_edit_mask(smiles: str, max_len: int = 120) -> torch.Tensor: return torch.ones(1, max_len)
def smiles_to_tensor(smiles_list, max_len=120): return torch.zeros(len(smiles_list), max_len, dtype=torch.long)

class MultimodalDataset(InMemoryDataset):
    def __init__(self, root, dataset, xd, xt, y, drug_fp, 
                 drug_id, rna_id, smile_graph, transform=None, pre_transform=None, 
                 augment=False, y_stats=None, rna_embeddings=None, max_rna_len=401,
                 force_reprocess=False):
        self.dataset_name = dataset
        self.xd = xd  
        self.xt = xt  
        self.y = y    
        self.drug_fp = drug_fp
        self.drug_id = drug_id
        self.rna_id = rna_id
        self.smile_graph = smile_graph
        self.augment = augment 
        self.y_stats = y_stats  
        self.rna_embeddings = rna_embeddings
        self.max_rna_len = max_rna_len
        if force_reprocess:
            processed_path = os.path.join(root, "processed", f"{dataset}.pt")
            if os.path.exists(processed_path):
                os.remove(processed_path)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return [f'{self.dataset_name}.pt']
    
    def _save_stats(self, y_min, y_max):
        stats_dir = os.path.join(self.root, "processed", "stats")
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, f"{self.dataset_name}_stats.pt")
        torch.save((y_min, y_max), stats_path)
        print(f"[Info] Saved stats to {stats_path}")

    def download(self): pass

    def process(self):
        labels = np.array(self.y, dtype=np.float32)
        if self.y_stats is not None:
            y_min, y_max = self.y_stats
        else:
            y_min, y_max = labels.min(), labels.max()
            
        denominator = (y_max - y_min) if (y_max - y_min) > 1e-9 else 1.0
        normalized_labels = (labels - y_min) / denominator

        if self.y_stats is None: self._save_stats(y_min, y_max)

        data_list = []
        data_len = len(self.xd)
        for i in range(data_len):
            smiles = self.xd[i]
            target = normalized_labels[i]
            rna_seq = self.xt[i]
            
            if smiles in self.smile_graph:
                x, edge_index, edge_attr = self.smile_graph[smiles]
            else: continue
            
            c_size = x.size(0)
            data = Data(
                x=torch.FloatTensor(x),
                edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.FloatTensor(edge_attr),
                y=torch.FloatTensor([target])
            )
            
            data.rna_seq = str(rna_seq)       
            data.smiles_seq = str(smiles)     
            data.drug_fp = torch.FloatTensor([self.drug_fp[i]])
            data.__num_nodes__ = int(c_size) 
            
            padded_emb = torch.zeros((self.max_rna_len, 640))
            rna_mask = torch.zeros(self.max_rna_len)
            
            if self.rna_embeddings is not None and self.rna_id is not None:
                r_id = self.rna_id[i]
                if r_id in self.rna_embeddings:
                    raw_emb = torch.tensor(self.rna_embeddings[r_id], dtype=torch.float)
                    seq_len = min(raw_emb.size(0), self.max_rna_len)
                    
                    padded_emb[:seq_len, :] = raw_emb[:seq_len, :]
                    rna_mask[:seq_len] = 1.0
            
            data.rna_embedding = padded_emb.unsqueeze(0)
            data.rna_mask = rna_mask.unsqueeze(0)
            
            if self.rna_id: data.rna_id = self.rna_id[i]
            if self.drug_id: data.drug_id = self.drug_id[i]

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[Info] Saved processed dataset (Padded Sequence Mode): {self.processed_paths[0]} (Max_Len={self.max_rna_len})")

    def get(self, idx):
        data = super().get(idx).clone()
        if self.augment and random.random() < 0.5:
            original_smiles = data.smiles_seq
            try:
                mol = Chem.MolFromSmiles(original_smiles)
                if mol is not None:
                    new_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                    data.smiles_seq = new_smiles
            except: pass
        return data

def get_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    pearson = 0.0 if np.std(y_true) == 0 or np.std(y_pred) == 0 else float(stats.pearsonr(y_true, y_pred)[0])
    spearman = 0.0 if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1 else float(stats.spearmanr(y_true, y_pred)[0])
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Pearson': pearson, 'Spearman': spearman}
