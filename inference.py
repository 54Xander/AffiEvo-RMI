import torch
import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
import itertools
import re

from models.predictor import AffinityPredictor
from models.optimizer import EvolutionaryOptimizer
from utils.chem_utils import (
    set_seed, molecule_quality_metrics, is_optimization_success
)

from create_data import extract_rna_features


INFERENCE_FOLD = 1
_RNA_FM_RUNTIME = {}
_RNA_SEQUENCE_CACHE = {}

def _get_rna_fm_runtime(device):
    from transformers import EsmModel, EsmTokenizer, EsmConfig

    device_key = str(device)
    if device_key in _RNA_FM_RUNTIME:
        return _RNA_FM_RUNTIME[device_key]

    model_path = "./pre-train/RNA-FM"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到本地预训练模型路径 {model_path}")

    config = EsmConfig.from_pretrained(model_path)
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel(config)

    weight_file = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(weight_file):
        weight_file = os.path.join(model_path, "model.safetensors")
    state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    new_state_dict = {
        k.replace("rnafm.", "").replace("layer_norm", "LayerNorm"): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()
    _RNA_FM_RUNTIME[device_key] = (tokenizer, model)
    return _RNA_FM_RUNTIME[device_key]


def get_rna_emb_patch(rna_seq, device):
    normalized_seq = str(rna_seq).strip().upper()
    cache_key = (str(device), normalized_seq)
    if cache_key in _RNA_SEQUENCE_CACHE:
        return _RNA_SEQUENCE_CACHE[cache_key]
    try:
        tokenizer, model = _get_rna_fm_runtime(device)

        with torch.no_grad():
            inputs = tokenizer([normalized_seq], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state.cpu().numpy().astype(np.float32)
            embedding = last_hidden[0]
            _RNA_SEQUENCE_CACHE[cache_key] = embedding
            return embedding

    except Exception as e:
        print(f"  ❌ 实时提取特征失败: {e}")
        return None


def _cached_embeddings_by_sequence(dataset, rna_emb_dict):
    if not isinstance(rna_emb_dict, dict):
        return {}
    rna_path = os.path.join("data", dataset, "RNA.xlsx")
    if not os.path.exists(rna_path):
        rna_path = os.path.join("data", "RNA-Molecule", "RNA.xlsx")
    if not os.path.exists(rna_path):
        return {}
    try:
        rna_df = pd.read_excel(rna_path)
    except Exception:
        return {}
    if "RNA_ID" not in rna_df or "RNA_Sequence" not in rna_df:
        return {}
    by_sequence = {}
    for _, row in rna_df[["RNA_ID", "RNA_Sequence"]].dropna().iterrows():
        emb = rna_emb_dict.get(row["RNA_ID"], rna_emb_dict.get(str(row["RNA_ID"])))
        if emb is not None:
            by_sequence.setdefault(str(row["RNA_Sequence"]).strip().upper(), emb)
    return by_sequence


def validate_checkpoint_identity(model_path, checkpoint, split_mode, fold=INFERENCE_FOLD):
    filename = os.path.basename(os.fspath(model_path))
    fold_matches = re.findall(r"(?:^|[_-])fold(\d+)(?:[_\.-]|$)", filename, flags=re.IGNORECASE)
    if fold_matches and any(int(value) != fold for value in fold_matches):
        raise ValueError(
            f"Checkpoint fold mismatch: inference requires fold{fold}, got {filename}"
        )
    if not fold_matches:
        print(f"⚠️ checkpoint 文件名未包含 fold 标记，无法从文件名确认 fold{fold}: {filename}")

    metadata = checkpoint.get("checkpoint_meta", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict) and isinstance(metadata, dict):
        metadata = dict(metadata)
        for key in ("fold", "split_mode"):
            if key in checkpoint and key not in metadata:
                metadata[key] = checkpoint[key]
    if isinstance(metadata, dict):
        ckpt_fold = metadata.get("fold")
        if ckpt_fold is not None and int(ckpt_fold) != fold:
            raise ValueError(f"Checkpoint metadata fold={ckpt_fold} 与 inference fold={fold} 不一致")
        ckpt_split = metadata.get("split_mode")
        if ckpt_split is not None and str(ckpt_split) != str(split_mode):
            raise ValueError(
                f"Checkpoint metadata split_mode={ckpt_split!r} 与 inference split_mode={split_mode!r} 不一致"
            )

    known_splits = ("Molecule_Scaffold", "random")
    stem = os.path.splitext(filename)[0]
    filename_split = next(
        (name for name in sorted(known_splits, key=len, reverse=True)
         if stem.lower().startswith(name.lower() + "_")),
        None,
    )
    if filename_split and filename_split.lower() != str(split_mode).lower():
        raise ValueError(
            f"Checkpoint split marker {filename_split!r} 与 inference split_mode={split_mode!r} 不一致"
        )

def load_model_native_stats(dataset, fold=INFERENCE_FOLD, split_mode="random"):
    path_fold = os.path.join(
        "data", "processed", f"{dataset}_{split_mode}_fold{fold}_train_y_stats.pt"
    )
    if not os.path.exists(path_fold):
        raise FileNotFoundError(
            f"Missing split-specific training stats: {path_fold}. "
            "Regenerate data before inference."
        )

    print(f"📖 加载统计参数: {path_fold}")
    stats = torch.load(path_fold, map_location='cpu', weights_only=True)
    return float(stats[0]), float(stats[1])

def run_batch_inference(args):
    run_optimization = args.mode == "optimize"

    args_dict = vars(args).copy()
    args_dict["do_optimize"] = int(run_optimization)

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"🚦 运行模式: {'预测 + 结构优化 (optimize)' if run_optimization else '仅预测 (predict)'}")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"❌ 找不到输入文件: {args.input_file}")

    print(f"📂 读取输入文件: {args.input_file}")
    df = pd.read_csv(args.input_file) if args.input_file.endswith(".csv") else pd.read_excel(args.input_file)

    col_map = {col.lower().replace(" ", "_"): col for col in df.columns}
    smi_col = next((col_map[p] for p in ["smiles", "smi", "molecule"] if p in col_map), None)
    rna_col = next((col_map[p] for p in ["rna_sequence", "rna_seq", "sequence"] if p in col_map), None)

    if not smi_col or not rna_col:
        raise ValueError(
            f"❌ 缺少 SMILES 或 RNA_Sequence 列。当前列: {list(df.columns)}"
        )

    df = df.rename(columns={smi_col: "smiles", rna_col: "rna_sequence"})
    print(f"📊 总计 {len(df)} 条数据。")

    actual_y_min, actual_y_max = load_model_native_stats(args.dataset, fold=INFERENCE_FOLD, split_mode=args.split_mode)
    y_range = actual_y_max - actual_y_min

    print("🔄 正在从缓存加载 RNA 特征...")
    rna_emb_dict = extract_rna_features(args.dataset, force_recalc=False)
    cached_by_sequence = _cached_embeddings_by_sequence(args.dataset, rna_emb_dict)

    predictor = AffinityPredictor(
        dim_atom=args.dim_atom,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        args=args_dict,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    validate_checkpoint_identity(args.model_path, checkpoint, args.split_mode, fold=INFERENCE_FOLD)
    predictor.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    predictor.eval()

    optimizer_tool = None
    if run_optimization:
        optimizer_tool = EvolutionaryOptimizer(predictor, device, actual_y_min, actual_y_max, args_dict)

    results = []

    for idx, row in df.iterrows():
        smi = str(row["smiles"]).strip()
        rna_str = str(row["rna_sequence"]).strip()

        if not smi or not rna_str or smi.lower() == "nan" or rna_str.lower() == "nan":
            print(f"⚠️ [{idx + 1}/{len(df)}] 跳过空 SMILES 或 RNA 序列")
            continue

        normalized_rna = rna_str.upper()
        rna_feature = cached_by_sequence.get(normalized_rna)

        if rna_feature is None:
            rna_feature = get_rna_emb_patch(rna_str, device)

        if rna_feature is None:
            print(f"⚠️ [{idx + 1}/{len(df)}] RNA 特征提取失败，跳过该条数据")
            continue

        rna_tensor = torch.from_numpy(rna_feature).float().to(device)

        if rna_tensor.dim() == 1:
            rna_tensor = rna_tensor.unsqueeze(0)

        if rna_tensor.dim() != 2:
            raise ValueError(
                f"RNA embedding for row {idx} must be [L, D], "
                f"got {tuple(rna_tensor.shape)}"
            )

        rna_tensor = rna_tensor[: args.max_rna_len, :]

        with torch.no_grad():
            initial_output = predictor.predict_smiles_list(
                [smi],
                rna_embedding=rna_tensor,
            )
            init_z = (
                initial_output[0].item()
                if hasattr(initial_output, "__len__")
                else initial_output.item()
            )
            init_real = init_z * y_range + actual_y_min

        init_qed, init_sa, _ = molecule_quality_metrics(smi)

        init_pKd_value = round(init_real, 4)
        init_qed_value = round(init_qed, 4) if np.isfinite(init_qed) else np.nan
        init_sa_value = round(init_sa, 4) if np.isfinite(init_sa) else np.nan

        if not run_optimization:
            row_result = {
                "original_smiles": smi,
                "rna_sequence": rna_str,
                "Init_pKd": init_pKd_value,
                "Init_QED": init_qed_value,
                "Init_SA": init_sa_value,
            }
            results.append(row_result)

            print(f"[{idx + 1}/{len(df)}] 预测 pKd: {init_real:.3f}")
            continue

        row_result = {
            "original_smiles": smi,
            "rna_sequence": rna_str,
            "optimized_smiles": smi,
            "Init_pKd": init_pKd_value,
            "Final_pKd": init_pKd_value,
            "ΔpKd": 0.0,

            "Init_QED": init_qed_value,
            "Final_QED": init_qed_value,

            "Init_SA": init_sa_value,
            "Final_SA": init_sa_value,

            "Tanimoto similarity": 1.0,
            "Optimization_Success": False,
            "Best_Composite_SMILES": smi,
            "Successful_Optimized_SMILES": "",
        }

        best_smi, history = optimizer_tool.optimize(
            smi,
            rna_tensor,
            args.opt_steps,
            args.opt_candidates,
            similarity_cutoff=args.sim_cutoff,
            success_delta=args.success_delta,
        )

        if history:
            final_real = history[-1][1]

            final_qed, final_sa, tanimoto = molecule_quality_metrics(
                best_smi,
                reference_smiles=smi,
            )

            success = is_optimization_success(
                final_real - init_real,
                final_sa,
                tanimoto,
                delta_threshold=args.success_delta,
                sa_cutoff=args.sa_cutoff,
                similarity_cutoff=args.sim_cutoff,
            )

            best_composite_smi = getattr(optimizer_tool, "last_best_composite_smiles", best_smi)

            row_result.update(
                {
                    "optimized_smiles": best_smi,
                    "Final_pKd": round(final_real, 4),
                    "ΔpKd": round(final_real - init_real, 4),
                    "Final_QED": (
                        round(final_qed, 4)
                        if np.isfinite(final_qed)
                        else np.nan
                    ),
                    "Final_SA": (
                        round(final_sa, 4)
                        if np.isfinite(final_sa)
                        else np.nan
                    ),
                    "Tanimoto similarity": (
                        round(tanimoto, 4)
                        if np.isfinite(tanimoto)
                        else np.nan
                    ),
                    "Optimization_Success": success,
                    "Best_Composite_SMILES": best_composite_smi,
                    "Successful_Optimized_SMILES": (
                        best_smi if success else ""
                    ),
                }
            )
        else:
            print(
                f"⚠️ [{idx + 1}/{len(df)}] "
                "优化器未返回有效 history，保留原始分子作为结果"
            )

        results.append(row_result)

        print(f"[{idx + 1}/{len(df)}] 初始 pKd: {init_real:.3f} | 优化后 pKd: {row_result['Final_pKd']:.3f} | "
              f"ΔpKd: {row_result['ΔpKd']:.3f} | Success: {row_result['Optimization_Success']}")

    output_df = pd.DataFrame(results)

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    output_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")

    print(f"🎉 完成！保存至: {args.output_file}")

    if run_optimization:
        print("📌 输出模式: 预测 + 结构优化；已输出 Init/Final/ΔpKd 及优化相关指标。")
    else:
        print("📌 输出模式: 仅预测；未执行结构优化，因此不输出 Final/ΔpKd 等优化字段。")


def get_args():
    parser = argparse.ArgumentParser(description="RNA-小分子亲和力预测 / 结构优化推理脚本")

    parser.add_argument("--model_path", type=str, default="results/models/random_best_predictor_fold1.pt")
    parser.add_argument("--input_file", type=str, default="predict/test_data.csv")
    parser.add_argument("--output_file", type=str, default="predict/results_inference.csv")
    parser.add_argument("--dataset", type=str, default="RNA-Molecule")
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "Molecule_Scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", )

    parser.add_argument("--mode", type=str, default="optimize", choices=["predict", "optimize"],
        help=("运行模式："
              "'predict' = 只预测原始分子的 pKd/QED/SA；"
              "'optimize' = 预测后继续进行结构优化。"
        ),
    )

    parser.add_argument("--dim_atom", type=int, default=80)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.8)
    parser.add_argument("--max_rna_len", type=int, default=401)
    parser.add_argument("--max_smiles_len", type=int, default=350)

    parser.add_argument("--opt_steps", type=int, default=50)
    parser.add_argument("--opt_candidates", type=int, default=80)
    parser.add_argument("--sim_cutoff", type=float, default=0.1, help="Minimum lead-preservation Tanimoto similarity")
    parser.add_argument("--sa_weight", type=float, default=0.08)
    parser.add_argument("--sa_cutoff", type=float, default=5.5, help="Hard SA constraint used for optimization success")
    parser.add_argument("--success_delta", type=float, default=0.3, help="Minimum predicted pKd improvement for success")
    parser.add_argument("--qed_weight", type=float, default=1.5)
    parser.add_argument("--mutation_probs", type=float, nargs=3, default=[0.35, 0.60, 0.80])
    parser.add_argument("--aromatic_protect_prob", type=float, default=0.8,
                        help="Probability of skipping mutation when the selected atom is aromatic")

    parser.add_argument("--no_interaction", action="store_true")
    parser.add_argument("--no_gnn", action="store_true")
    parser.add_argument("--no_fp", action="store_true")
    parser.add_argument("--no_masking", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_batch_inference(get_args())
