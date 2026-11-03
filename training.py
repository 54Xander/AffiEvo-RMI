import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger
import torch.nn.functional as F
from datetime import datetime
import random
import gc
import csv

from models.predictor import AffinityPredictor
from models.optimizer import EvolutionaryOptimizer
from utils.chem_utils import (
    set_seed, molecule_quality_metrics, is_optimization_success
)
from utils.train_metrics import (
    ensure_dir, save_metrics_to_csv,
    evaluate_predictor,
    load_single_fold_data
)

from create_data import prepare_all_data, load_fold_train_stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
os.environ['PYTHONHASHSEED'] = str(42)

torch.use_deterministic_algorithms(True, warn_only=True) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

RDLogger.DisableLog('rdApp.*')


class MultiTaskLoss(nn.Module):
    def __init__(self, reg_weight=1.0, rank_weight=0.2, aux_weight=0.1):
        super().__init__()
        self.register_buffer('weights', torch.tensor([reg_weight, rank_weight, aux_weight]))

    def forward(self, mse_loss, rank_loss, aux_loss):
        final_loss = (self.weights[0] * mse_loss + 
                      self.weights[1] * rank_loss + 
                      self.weights[2] * aux_loss)
        return final_loss


def resolve_aux_loss_weights(args):
    aux_weight = args.get('aux_weight')
    if aux_weight is None:
        aux_weight = args.get('cl_weight', 0.005)

    alignment_weight = args.get('alignment_weight', 0.1)
    hard_mining_weight = args.get('hard_mining_weight', 1.0)

    if args.get('no_aux', False):
        aux_weight = 0.0
    if args.get('no_alignment', False) or args.get('no_aux', False):
        alignment_weight = 0.0
    if args.get('no_hard_mining', False) or args.get('no_aux', False):
        hard_mining_weight = 0.0

    return float(aux_weight), float(alignment_weight), float(hard_mining_weight)

def alignment_loss(mol_vec, rna_vec, labels, threshold=0.7):
    cos_sim = F.cosine_similarity(mol_vec, rna_vec)
    high_affinity_mask = (labels > threshold).float()
    loss_align = (1 - cos_sim) * high_affinity_mask
    return loss_align.mean()

def hybrid_loss_components(pred, target, margin=0.1):
    mse_loss = nn.HuberLoss(delta=1.0)(pred, target)
    
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    target_sign = torch.sign(target_diff)
    pair_mask = (torch.abs(target_diff) > 1e-6).float()
    
    rank_loss = torch.relu(-target_sign * pred_diff + margin)
    rank_loss = (rank_loss * pair_mask).sum() / (pair_mask.sum() + 1e-9)
    
    return mse_loss, rank_loss

def contrastive_loss_hard_mining(mol_feats, labels, alpha=2.0, temp=1.0):
    mol_feats = F.normalize(mol_feats, p=2, dim=1)
    dist_matrix = torch.cdist(mol_feats, mol_feats, p=2)
    label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
    
    if temp != 1.0:
        dist_matrix = dist_matrix / temp
    
    weight_matrix = torch.pow(label_diff + 1.0, alpha)
    loss_matrix = F.smooth_l1_loss(dist_matrix, label_diff, reduction='none')
    
    loss = (loss_matrix * weight_matrix).mean()
    return loss


def train_predictor_epoch_custom(model, mtl_module, loader, optimizer, device, args, **kwargs):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    margin = kwargs.get('margin', args['margin'])
    cl_alpha = kwargs.get('cl_alpha', args['cl_alpha'])
    _, alignment_weight, hard_mining_weight = resolve_aux_loss_weights(args)
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        
        if isinstance(out, tuple):
            pred, mol_features = out
        else:
            pred, mol_features = out, None
        target = batch.y.view(-1)
        
        mse_l, rank_l = hybrid_loss_components(pred, target, margin=margin)
        
        aux_l = torch.tensor(0.0).to(device)
        if (mol_features is not None and len(mol_features) == 2 and
                mol_features[0] is not None and mol_features[1] is not None):
            rna_vec, mol_vec = mol_features
            
            align_l = alignment_loss(rna_vec, mol_vec, target)
            cl_l = contrastive_loss_hard_mining(mol_vec, target, alpha=cl_alpha)
            aux_l = alignment_weight * align_l + hard_mining_weight * cl_l
        
        loss = mtl_module(mse_l, rank_l, aux_l)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.detach().cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 0
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)), 
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred), 
        'Pearson': pearson, 
        'Spearman': spearman
    }
    return avg_loss, metrics

def get_optimizer(model, mtl_module, args):
    params = [
        {'params': model.parameters(), 'lr': args['lr']}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args['weight_decay'])
    print(f"[Optimizer] Model parameters included. MTL Weights are FIXED.")
    return optimizer


def run_single_fold_training(fold, args, tra_loader, val_loader, tes_loader, vis_range, g_min, g_max, best_model_path, evaluate_test=True):
    set_seed(args['seed'])    
    device = args['device']
    predictor = AffinityPredictor(
        dim_atom=args['dim_atom'],
        embed_dim=args['embed_dim'],
        num_heads=args['num_heads'],
        gnn_layers=args['gnn_layers'],
        dropout=args['dropout'],
        args=args
    ).to(device)
    
    aux_weight, alignment_weight, hard_mining_weight = resolve_aux_loss_weights(args)
    mtl_module = MultiTaskLoss(
        reg_weight=args['reg_weight'], 
        rank_weight=args['rank_weight'], 
        aux_weight=aux_weight
    ).to(device)
    
    optimizer = get_optimizer(predictor, mtl_module, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=args['patience'])
    current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictor_result_dir = os.path.join("results", "predictor")
    ensure_dir(predictor_result_dir)

    best_val_pearson = -1.0
    best_epoch = -1
    patience_counter = 0
    early_stop_patience = args['early_stop']

    print("Epoch |Loss | RMSE(Real) MAE(Real) R2 Pea Spea ")

    set_seed(args['seed'])
    for epoch in range(args['epochs']):
        
        train_loss, train_metrics = train_predictor_epoch_custom(
            predictor, mtl_module, tra_loader, optimizer, device, args,
            cl_alpha=args['cl_alpha'], margin=args['margin']
        )
        
        val_loss, val_metrics = evaluate_predictor(predictor, val_loader, device, args, fold)
        scheduler.step(val_metrics['Pearson'])

        for m_dict, raw_loss in zip([train_metrics, val_metrics], [train_loss, val_loss]):
            m_dict['Real_RMSE'] = m_dict['RMSE'] * vis_range
            m_dict['Real_MAE'] = m_dict['MAE'] * vis_range
            m_dict['Real_Loss'] = raw_loss * (vis_range**2)

        weights = mtl_module.weights.cpu().numpy()
        print(
            f"[{epoch:03d}] Weights: MSE={weights[0]:.2f}, "
            f"Rank={weights[1]:.2f}, Aux={weights[2]:.3f} "
            f"(Alignment={alignment_weight:.3f}, HardMining={hard_mining_weight:.3f})"
        )
        
        def get_log_str(loss, m):
            return f"Loss: {loss:.4f} | {m['RMSE']:.4f} ({m['Real_RMSE']:.4f}) {m['MAE']:.4f} ({m['Real_MAE']:.4f}) {m['R2']:.4f} {m['Pearson']:.4f} {m['Spearman']:.4f}"

        print(f"Tra: {get_log_str(train_loss, train_metrics)}")
        print(f"Val: {get_log_str(val_loss, val_metrics)}")

        
        save_metrics_to_csv(
            train_metrics, predictor_result_dir, "predictor", epoch, fold,
            args['split_mode'], "train", {"loss": train_loss},
            run_id=current_run_id
        )
        
        save_metrics_to_csv(
            val_metrics, predictor_result_dir, "predictor", epoch, fold,
            args['split_mode'], "val", {"loss": val_loss},
            run_id=current_run_id
        )

        current_val_pearson = val_metrics['Pearson']
        
        if current_val_pearson > best_val_pearson:
            best_val_pearson = current_val_pearson
            best_epoch = epoch
            ensure_dir(os.path.dirname(best_model_path))
            torch.save({
                "model_state_dict": predictor.state_dict(),
                "checkpoint_meta": {
                    "dataset": args.get("dataset"),
                    "split_mode": args.get("split_mode", "random"),
                    "fold": int(fold),
                    "best_epoch": int(epoch),
                },
            }, best_model_path)
            print(f">>> New Best Model!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f">>> No improvement for {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(f"[Early Stopping] Fold {fold} triggered.")
                break
    
    if best_epoch < 0:
        raise RuntimeError(f"Fold {fold} produced no checkpoint; cannot run final test evaluation")
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    predictor.load_state_dict(
        best_checkpoint.get("model_state_dict", best_checkpoint), strict=True
    )
    if not evaluate_test:
        return {"best_epoch": best_epoch, "val_pearson": best_val_pearson}

    test_loss, best_test_metrics = evaluate_predictor(
        predictor, tes_loader, device, args, fold
    )
    best_test_metrics['Real_RMSE'] = best_test_metrics['RMSE'] * vis_range
    best_test_metrics['Real_MAE'] = best_test_metrics['MAE'] * vis_range
    best_test_metrics['Real_Loss'] = test_loss * (vis_range ** 2)
    
    save_metrics_to_csv(
        best_test_metrics, predictor_result_dir, "predictor", best_epoch, fold,
        args['split_mode'], "test", {"loss": test_loss},
        run_id=current_run_id
    )

    
    
    
    
    print(
        f"Fold {fold} Best Epoch: {best_epoch} | Final Test Pearson: "
        f"{best_test_metrics['Pearson']:.4f} | Final Test RMSE: "
        f"{best_test_metrics['Real_RMSE']:.4f}"
    )
    return {"best_epoch": best_epoch, "val_pearson": best_val_pearson,
            "test_metrics": best_test_metrics}

def run_single_fold_optimization(fold, args, predictor, tes_loader, vis_range, g_min, g_max):
    set_seed(args['seed'])
    device = args['device']
    predictor.eval()

    optimizer_result_dir = os.path.join("results", "optimizer")
    ensure_dir(optimizer_result_dir)

    optimizer_tool = EvolutionaryOptimizer(
        predictor_model=predictor,
        device=device,
        y_min=g_min,
        y_max=g_max,
        mutation_probs=args['mutation_probs'],
        qed_weight=args.get('qed_weight', 2.0),
        sa_weight=args.get('sa_weight', 0.2),
        sa_cutoff=args.get('sa_cutoff', 5.5),
        aromatic_protect_prob=args.get('aromatic_protect_prob', 0.8)
    )

    current_run_id = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = os.path.join(
        optimizer_result_dir,
        f"{args['split_mode']}_fold{fold}_optimized_{current_run_id}.csv"
    )

    with open(output_file, "w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "RNA_seq",
            "Init_SMILES",
            "Final_SMILES",
            "Init_Pred_Affinity",
            "Final_Pred_Affinity",
            "Affinity_Improvement",
            "Init_pKd",
            "Final_pKd",
            "ΔpKd",
            "Init_QED",
            "Final_QED",
            "Init_SA",
            "Final_SA",
            "Tanimoto similarity",
            "Optimization_Success",
            "Best_Composite_SMILES",
            "Successful_Optimized_SMILES"
        ])

    target_subject_count = 10

    cohort_by_rna = {}

    for test_batch in tes_loader:
        test_batch = test_batch.to(device)

        for i in range(len(test_batch.smiles_seq)):
            target_rna_seq = str(test_batch.rna_seq[i])
            init_smi = str(test_batch.smiles_seq[i])
            target_rna_emb = test_batch.rna_embedding[i]
            target_rna_mask = getattr(test_batch, "rna_mask", None)
            if target_rna_mask is not None:
                target_rna_mask = target_rna_mask[i]

            try:
                if target_rna_mask is None:
                    raise ValueError("Missing rna_mask in optimization batch")

                while target_rna_emb.dim() > 2 and target_rna_emb.size(0) == 1:
                    target_rna_emb = target_rna_emb.squeeze(0)
                while target_rna_mask.dim() > 1 and target_rna_mask.size(0) == 1:
                    target_rna_mask = target_rna_mask.squeeze(0)

                if target_rna_emb.dim() != 2 or target_rna_mask.dim() != 1:
                    raise ValueError(
                        "Unexpected RNA embedding/mask shapes: "
                        f"{tuple(target_rna_emb.shape)}, {tuple(target_rna_mask.shape)}"
                    )

                valid_len = int(target_rna_mask.bool().sum().item())
                if valid_len <= 0:
                    raise ValueError("RNA mask contains no valid tokens")

                trimmed_emb = target_rna_emb[:valid_len].detach().cpu().clone()
            except Exception as exc:
                print(
                    f"[Fold {fold} Cohort skip] RNA={target_rna_seq!r} "
                    f"SMILES={init_smi!r} | {type(exc).__name__}: {exc}"
                )
                continue

            candidate = (target_rna_seq, init_smi, trimmed_emb)
            previous = cohort_by_rna.get(target_rna_seq)

            if previous is None or init_smi < previous[1]:
                cohort_by_rna[target_rna_seq] = candidate

    cohort = [
        cohort_by_rna[rna_key]
        for rna_key in sorted(cohort_by_rna)
    ][:target_subject_count]

    print(
        f"[Fold {fold} Cohort] fixed_subjects={len(cohort)}/"
        f"{target_subject_count} | unique_valid_rnas={len(cohort_by_rna)}"
    )

    if not cohort:
        raise RuntimeError(f"Fold {fold} produced no valid optimization subjects")

    completed_count = 0
    failed_count = 0

    for subject_idx, (target_rna_seq, init_smi, emb_cpu) in enumerate(cohort, start=1):
        subject_seed = int(args['seed']) + int(fold) * 1000 + int(subject_idx)
        set_seed(subject_seed)
        target_rna_emb = emb_cpu.to(device)

        init_qed, init_sa, _ = molecule_quality_metrics(init_smi)
        init_real = float("nan")
        final_real = float("nan")
        final_qed = init_qed
        final_sa = init_sa
        tanimoto = 1.0
        best_mol = init_smi
        best_composite_mol = init_smi
        success = False
        successful_smi = ""

        try:
            init_real = optimizer_tool._predict_single(init_smi, target_rna_emb)

            best_mol, history = optimizer_tool.optimize(
                initial_smiles=init_smi,
                rna_embedding=target_rna_emb,
                steps=args['opt_steps'],
                population_size=args['opt_candidates'],
                similarity_cutoff=args.get('sim_cutoff', 0.2),
                success_delta=args.get('success_delta', 0.3)
            )

            if not history:
                raise RuntimeError("optimizer returned empty history")

            history_init = history[0][1]
            if np.isfinite(init_real) and np.isfinite(history_init):
                if abs(float(history_init) - float(init_real)) > 1e-6:
                    raise RuntimeError(
                        "Initial affinity changed inside optimizer: "
                        f"pre={init_real:.8f}, history={float(history_init):.8f}"
                    )

            final_real = history[-1][1]
            final_qed, final_sa, tanimoto = molecule_quality_metrics(
                best_mol, reference_smiles=init_smi
            )
            best_composite_mol = getattr(
                optimizer_tool, "last_best_composite_smiles", best_mol
            )

            success = is_optimization_success(
                final_real - init_real,
                final_sa,
                tanimoto,
                delta_threshold=args.get('success_delta', 0.3),
                sa_cutoff=args.get('sa_cutoff', 5.5),
                similarity_cutoff=args.get('sim_cutoff', 0.2)
            )
            successful_smi = best_mol if success else ""

        except Exception as exc:
            failed_count += 1
            print(
                f"[Fold {fold} Opt fallback {subject_idx}/{len(cohort)}] "
                f"RNA={target_rna_seq!r} SMILES={init_smi!r} | "
                f"{type(exc).__name__}: {exc}"
            )

            if not np.isfinite(init_real):
                try:
                    init_real = optimizer_tool._predict_single(
                        init_smi, target_rna_emb
                    )
                except Exception:
                    init_real = float("nan")

            final_real = init_real
            final_qed = init_qed
            final_sa = init_sa
            tanimoto = 1.0
            best_mol = init_smi
            best_composite_mol = init_smi
            success = False
            successful_smi = ""

        with open(output_file, "a", newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                target_rna_seq,
                init_smi,
                best_mol,
                init_real,
                final_real,
                final_real - init_real if np.isfinite(init_real) and np.isfinite(final_real) else float("nan"),
                init_real,
                final_real,
                final_real - init_real if np.isfinite(init_real) and np.isfinite(final_real) else float("nan"),
                init_qed,
                final_qed,
                init_sa,
                final_sa,
                tanimoto,
                success,
                best_composite_mol,
                successful_smi
            ])

        completed_count += 1
        improvement = (
            final_real - init_real
            if np.isfinite(init_real) and np.isfinite(final_real)
            else float("nan")
        )
        print(
            f"[Fold {fold} Opt {completed_count}/{len(cohort)}] "
            f"Seed={subject_seed} | Improvement={improvement:.4f} | "
            f"Success={success}"
        )

    print(
        f"[Fold {fold} Opt summary] fixed_subjects={len(cohort)}, "
        f"written={completed_count}, fallbacks={failed_count}"
    )


def get_ablation_tag(args):
    names = [name for name in (
        "no_interaction", "no_gnn", "no_fp", "no_masking",
        "no_alignment", "no_hard_mining", "no_aux"
    )
             if bool(args.get(name, False))]
    return "full" if not names else "_".join(names)


def run_training(args):
    device = args['device']
    ensure_dir("results/models")
    
    prepare_all_data(
        dataset=args['dataset'],
        n_splits=args['n_splits'],
        seed=args['seed'],
        val_size=args['val_size'],
        split_mode=args['split_mode'],
        max_rna_len=args['max_rna_len']
    )

    fold_results = []
    evaluate_test = bool(args.get('evaluate_test', True))
    for fold in range(1, args['n_splits'] + 1):
        set_seed(args['seed'])
        
        print(f"\n\n{'#'*20} STARTING FOLD {fold} {'#'*20}")
        tra_loader, val_loader, tes_loader = load_single_fold_data(args, fold)
        g_min, g_max = load_fold_train_stats(
            args['dataset'], fold, split_mode=args['split_mode']
        )
        g_min, g_max = float(g_min), float(g_max)
        vis_range = g_max - g_min
        if vis_range <= 1e-12:
            vis_range = 1.0
        ablation_tag = get_ablation_tag(args)
        model_name = (f"{args['split_mode']}_best_predictor_fold{fold}.pt"
                      if ablation_tag == "full" else
                      f"{args['split_mode']}_{ablation_tag}_best_predictor_fold{fold}.pt")
        best_model_path = os.path.join("results/models", model_name)
        
        if args['stage'] in ['train', 'all']:
            fold_result = run_single_fold_training(
                fold, args, tra_loader, val_loader, tes_loader, vis_range,
                g_min, g_max, best_model_path, evaluate_test=evaluate_test
            )
            fold_results.append(fold_result)
            gc.collect()
            torch.cuda.empty_cache()
        
        if args['stage'] in ['opt', 'all']:
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(
                    f"Missing trained checkpoint: {best_model_path}"
                )
            print(f"[Info] Loading model from {best_model_path} for optimization (consistency mode)...")
            predictor = AffinityPredictor(
                dim_atom=args['dim_atom'],
                embed_dim=args['embed_dim'],
                num_heads=args['num_heads'],
                gnn_layers=args['gnn_layers'],
                dropout=args['dropout'],
                args=args
            ).to(device)
            
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            predictor.load_state_dict(
                checkpoint.get("model_state_dict", checkpoint), strict=True
            )
            
            run_single_fold_optimization(fold, args, predictor, tes_loader, vis_range, g_min, g_max)

        print(f"Cleaning up Fold {fold}...")
        if 'predictor' in locals():
            del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2)

    if not evaluate_test and fold_results:
        val_scores = [r["val_pearson"] for r in fold_results if r is not None]
        return {"val_pearson_mean": float(np.mean(val_scores)),
                "val_pearson_std": float(np.std(val_scores)),
                "fold_results": fold_results}
    return {"fold_results": fold_results}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='train', choices=['train', 'opt', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, default='RNA-Molecule')
    parser.add_argument('--split_mode', type=str, default='random',
                        choices=['random'])
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--max_rna_len', type=int, default=401)
    parser.add_argument('--max_smiles_len', type=int, default=350)
    parser.add_argument('--epochs', type=int, default=500) 
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)   
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--rank_weight', type=float, default=0.1)
    parser.add_argument('--cl_weight', type=float, default=0.005,
                        help='Legacy name for the outer auxiliary-task weight')
    parser.add_argument('--aux_weight', type=float, default=None,
                        help='Outer auxiliary-task weight; overrides --cl_weight')
    parser.add_argument('--alignment_weight', type=float, default=0.05,
                        help='Weight of the RNA-molecule alignment loss')
    parser.add_argument('--hard_mining_weight', type=float, default=2.0,
                        help='Weight of the hard-mining contrastive loss')
    parser.add_argument('--no_alignment', action='store_true',
                        help='Ablation: disable the alignment loss')
    parser.add_argument('--no_hard_mining', action='store_true',
                        help='Ablation: disable the hard-mining contrastive loss')
    parser.add_argument('--no_aux', action='store_true',
                        help='Ablation: disable the complete auxiliary task')
    parser.add_argument('--margin', type=float, default=0.01)
    parser.add_argument('--cl_alpha', type=float, default=0.5)
    parser.add_argument('--dim_atom', type=int, default=80) 
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--opt_steps', type=int, default=50)
    parser.add_argument('--opt_candidates', type=int, default=80)
    parser.add_argument('--sim_cutoff', type=float, default=0.1,
                        help='Minimum lead-preservation Tanimoto similarity')
    parser.add_argument('--qed_weight', type=float, default=1.5)
    parser.add_argument('--sa_weight', type=float, default=0.08)
    parser.add_argument('--sa_cutoff', type=float, default=5.5,
                        help='Hard SA constraint used for optimization success')
    parser.add_argument('--success_delta', type=float, default=0.3,
                        help='Minimum predicted pKd improvement for success')
    parser.add_argument('--mutation_probs', type=float, nargs=3, default=[0.35, 0.60, 0.80])
    parser.add_argument('--aromatic_protect_prob', type=float, default=0.8,
                        help='Probability of skipping mutation when the selected atom is aromatic')
    parser.add_argument('--no_interaction', action='store_true')
    parser.add_argument('--no_gnn', action='store_true')
    parser.add_argument('--no_fp', action='store_true')
    parser.add_argument('--no_masking', action='store_true')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = get_args()
    set_seed(args['seed'])
    run_training(args)
