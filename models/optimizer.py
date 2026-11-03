import torch
import numpy as np
import random
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, QED, DataStructs

try:
    from utils import sascorer
except ImportError:
    sascorer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolutionaryOptimizer:
    def __init__(self, predictor_model, device, y_min=0.0, y_max=12.0, args=None, **kwargs):
        self.predictor = predictor_model
        self.device = device
        self.predictor.eval()
        self.atom_pool = [6, 7, 8, 9, 16, 17]
        
        self.y_min = y_min
        self.y_max = y_max
        self.y_range = y_max - y_min
        
        self.args = args if args else {}
        for k, v in kwargs.items():
            self.args[k] = v
        
        self.mutation_probs = self.args.get('mutation_probs', [0.35, 0.60, 0.80])
        
        self.aromatic_protect_prob = self.args.get(
            'aromatic_protect_prob',
            0.8
        )
        
        self.qed_weight = self.args.get('qed_weight', 2.0)
        self.sa_weight = self.args.get('sa_weight', 0.3)
        self.sa_cutoff = self.args.get('sa_cutoff', 5.5)
        self.structural_similarity_penalty_weight = self.args.get(
            'structural_similarity_penalty_weight',
            self.args.get('sim_penalty_weight', 5.0)
        )
        self.sim_penalty_weight = self.structural_similarity_penalty_weight

    def _get_fp(self, mol):
        try: return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except: return None

    def _similarity(self, fp1, fp2):
        if fp1 is None or fp2 is None: return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def mutate(self, parent_mol):
        if parent_mol is None: return None
        mol = Chem.RWMol(parent_mol)
        
        try:
            op_idx = random.random()
            thresholds = self.mutation_probs
            
            atom_indices = [a.GetIdx() for a in mol.GetAtoms()]
            if not atom_indices: return None
            
            target_idx = random.choice(atom_indices)
            target_atom = mol.GetAtomWithIdx(target_idx)

            
            if (
                target_atom.GetIsAromatic()
                and random.random() < self.aromatic_protect_prob
            ):
                return parent_mol


            if op_idx < thresholds[0]: 
                if target_atom.GetDegree() < 4:
                    new_idx = mol.AddAtom(Chem.Atom(random.choice(self.atom_pool)))
                    mol.AddBond(target_idx, new_idx, Chem.BondType.SINGLE)
                    
            elif op_idx < thresholds[1]: 
                n = mol.GetNumAtoms()
                if n >= 4:
                    potential_partners = [i for i in range(n) if i != target_idx and not mol.GetBondBetweenAtoms(target_idx, i)]
                    if potential_partners:
                        partner_idx = random.choice(potential_partners)
                        mol.AddBond(target_idx, partner_idx, Chem.BondType.SINGLE)
                            
            elif op_idx < thresholds[2]: 
                if not target_atom.GetIsAromatic() and target_atom.GetDegree() <= 2: 
                    mol.ReplaceAtom(target_idx, Chem.Atom(random.choice([6, 7, 8, 9, 16])))
                    
            else:
                if target_atom.GetDegree() == 1:
                    mol.RemoveAtom(target_idx)

            try:
                Chem.SanitizeMol(mol)
                frags = Chem.GetMolFrags(mol, asMols=True)
                if len(frags) > 1:
                    mol = max(frags, key=lambda m: (m.GetNumAtoms(), Chem.MolToSmiles(m, canonical=True)))
                    mol = Chem.RWMol(mol)
                Chem.SanitizeMol(mol)
            except:
                return None

            res_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            if not res_smi or Chem.MolFromSmiles(res_smi) is None: return None
            return mol
        except: 
            return None

    def _predict_single(self, smiles, rna_embedding):
        if rna_embedding is None:
            raise RuntimeError(
                f"RNA embedding is missing for SMILES {smiles!r}"
            )
        try:
            with torch.no_grad():
                norm_preds = self.predictor.predict_smiles_list(smiles_list=[smiles], rna_embedding=rna_embedding)
                if not isinstance(norm_preds, torch.Tensor) or norm_preds.numel() != 1:
                    raise ValueError(
                        f"predictor returned unexpected output shape/type: "
                        f"{type(norm_preds).__name__}, "
                        f"{getattr(norm_preds, 'shape', None)}"
                    )
                norm_val = float(norm_preds.reshape(-1)[0].item())
                if not np.isfinite(norm_val):
                    raise ValueError(f"predictor returned non-finite normalized affinity: {norm_val}")
                if norm_val == -10.0:
                    raise RuntimeError("predictor returned failure sentinel -10.0")
                real_val = norm_val * self.y_range + self.y_min
                return real_val
        except Exception as exc:
            raise RuntimeError(
                f"Affinity prediction failed for SMILES {smiles!r}"
            ) from exc

    def _score_candidate(self, smiles, rna_emb, parent_fp, similarity_cutoff=0.3):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return -100.0, -10.0
        
        aff = self._predict_single(smiles, rna_emb)
        
        qed = QED.qed(mol)
        
        sa = 10.0
        if sascorer:
            try:
                sa = sascorer.calculateScore(mol)
            except: sa = 10.0
        
        if sa > self.sa_cutoff: return -100.0, aff

        current_fp = self._get_fp(mol)
        sim = self._similarity(parent_fp, current_fp)
        
        
            
        
        current_fp = self._get_fp(mol)
        sim = self._similarity(parent_fp, current_fp)
        
        if sim < similarity_cutoff:
            return -100.0, aff
        
        score = aff + (qed * self.qed_weight) - (sa * self.sa_weight)
        
        return score, aff

        

    def optimize(self, initial_smiles, rna_embedding, steps=20, population_size=50, similarity_cutoff=0.3, **kwargs):
        self.last_best_composite_smiles = initial_smiles

        parent_mol = Chem.MolFromSmiles(initial_smiles)
        if not parent_mol:
            return initial_smiles, []
        parent_fp = self._get_fp(parent_mol)
        
        init_aff = self._predict_single(initial_smiles, rna_embedding)
        init_score, _ = self._score_candidate(
            initial_smiles, rna_embedding, parent_fp, similarity_cutoff
        )
        best_score = init_score
        best_smiles = initial_smiles
        best_affinity = init_aff
        
        best_affinity_smiles = initial_smiles
        best_feasible_affinity = init_aff


        history = [(0, best_affinity, best_smiles)]
        mol_cache = {initial_smiles}
        current_pool = [initial_smiles]

        steps = kwargs.get('steps', steps)
        population_size = kwargs.get('population_size', population_size)
        success_delta = kwargs.get("success_delta", 0.3)

        for step in range(steps):
            next_gen_candidates = []
            
            for smi in current_pool:
                mol = Chem.MolFromSmiles(smi)
                if not mol: continue
                for _ in range(5): 
                    mutated = self.mutate(mol)
                    if mutated:
                        m_smi = Chem.MolToSmiles(mutated, isomericSmiles=False, canonical=True)
                        if m_smi not in mol_cache:
                            next_gen_candidates.append(m_smi)
                            mol_cache.add(m_smi)
            
            if not next_gen_candidates: break
            
            scored = []
            
            for cand in next_gen_candidates:
                score, aff = self._score_candidate(
                    cand,
                    rna_embedding,
                    parent_fp,
                    similarity_cutoff
                )
            
                scored.append((cand, aff, score))
            
                if score > -100.0 and aff > best_feasible_affinity:
                    best_feasible_affinity = aff
                    best_affinity_smiles = cand

            
            
            scored.sort(key=lambda x: (x[2], x[0]), reverse=True)
            
            if not scored: break
            
            generation_best = scored[0]
            if generation_best[2] > best_score:
                best_score = generation_best[2]
                best_smiles = generation_best[0]
                best_affinity = generation_best[1]
                logger.info(
                    f"Step {step+1}: Best Score={best_score:.4f} | "
                    f"Aff={best_affinity:.4f}"
                )
            
            current_pool = [x[0] for x in scored[:population_size]]
            history.append((step + 1, best_affinity, best_smiles))
            

        self.last_best_composite_smiles = best_smiles
        
        if best_feasible_affinity - init_aff >= success_delta:
            selected_smiles = best_affinity_smiles
            selected_affinity = best_feasible_affinity
        else:
            selected_smiles = best_smiles
            selected_affinity = best_affinity
        
        history[-1] = (
            history[-1][0],
            selected_affinity,
            selected_smiles
        )
        
        return selected_smiles, history

            
