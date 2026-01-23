# exp3/3_40_build_text_refusal_direction.py
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def build_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """使用 sklearn 的 roc_auc_score，如果没有就用简单实现."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # 简单 rank-based AUC (忽略 ties 的精细处理，足够用来选层)
        labels = labels.astype(int)
        pos = labels.sum()
        neg = len(labels) - pos
        if pos == 0 or neg == 0:
            return float("nan")
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(scores)) + 1.0  # 1-based rank
        sum_ranks_pos = ranks[labels == 1].sum()
        auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
        return float(auc)


def _get_num_hidden_layers_from_config(model) -> int:
    """
    尝试从 Qwen2.5-VL 的 config 中获取 transformer 层数。
    优先使用 config.num_hidden_layers，其次尝试 text_config.num_hidden_layers。
    """
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None and hasattr(cfg, "text_config"):
        num_layers = getattr(cfg.text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("无法从 model.config 中获取 num_hidden_layers，请检查模型配置。")
    return int(num_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp3_mechanism.yaml",
        help="exp3 配置文件",
    )
    parser.add_argument(
        "--behavior_labels",
        type=str,
        default="data/external/external_behavior_labels_model-qwen2_5_vl_7b.jsonl",
        help="3_30 的输出文件",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录（默认使用 config 里的 direction_storage_dir）",
    )
    parser.add_argument(
        "--max_refusal",
        type=int,
        default=2000,
        help="最多使用多少条 REFUSAL 样本",
    )
    parser.add_argument(
        "--max_other",
        type=int,
        default=2000,
        help="最多使用多少条 OTHER 样本",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("id", "Qwen/Qwen2.5-VL-7B-Instruct")

    # 注意：不在这里立刻解析 cand_layers，而是等模型加载完再根据 config 决定
    raw_layers_cfg = cfg.get("candidate_layers_for_text_direction", None)

    dir_root = args.out_dir or cfg.get("direction_storage_dir", "data/risk_directions_qwen2_5_vl_7b")
    dir_root = Path(dir_root)
    dir_root.mkdir(parents=True, exist_ok=True)

    # 1) 读取行为标签
    in_path = Path(args.behavior_labels)
    print(f"[INFO] 读取行为标签: {in_path}")

    prompts = []
    labels = []  # 1 = REFUSAL, 0 = OTHER

    n_refusal, n_other = 0, 0
    with in_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            ex = json.loads(line)
            label = ex.get("behavior_label_external", "OTHER")
            prompt = ex.get("prompt", "")
            if label == "REFUSAL":
                if n_refusal >= args.max_refusal:
                    continue
                prompts.append(prompt)
                labels.append(1)
                n_refusal += 1
            else:
                if n_other >= args.max_other:
                    continue
                prompts.append(prompt)
                labels.append(0)
                n_other += 1

    labels = np.array(labels, dtype=np.int32)
    print(f"[INFO] 选用 REFUSAL 样本: {n_refusal}, OTHER 样本: {n_other}, 总计: {len(labels)}")

    if len(labels) == 0 or n_refusal == 0 or n_other == 0:
        print("[ERROR] 没有足够的 REFUSAL / OTHER 样本，无法建立方向")
        return

    # 2) 加载模型，只做文本 forward
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 加载模型: {model_id} 到 {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    # 2.1) 根据 config / 模型确定要扫描的层
    # 支持三种情况：
    #  - candidate_layers_for_text_direction: "all" / "auto" → 扫描所有 transformer 层 (1..num_hidden_layers)
    #  - candidate_layers_for_text_direction: [10,14,...]   → 使用给定列表
    #  - 未配置该字段（None）                             → 也按 "all" 处理
    if isinstance(raw_layers_cfg, str) and raw_layers_cfg.lower() in {"all", "auto"}:
        num_layers = _get_num_hidden_layers_from_config(model)
        # hidden_states 的索引为 [0..num_layers]：0 = embedding，其余是各层输出
        # 这里仿照 HiddenDetect，只扫 transformer block 层，不扫 embedding，所以从 1 开始
        cand_layers = list(range(1, num_layers + 1))
        print(f"[INFO] candidate_layers_for_text_direction='all'，自动扫描全部 {num_layers} 个 transformer 层: {cand_layers}")
    elif raw_layers_cfg is None:
        # 如果没有配置，默认也扫描全部层
        num_layers = _get_num_hidden_layers_from_config(model)
        cand_layers = list(range(1, num_layers + 1))
        print(f"[INFO] 未在 config 中指定候选层，默认扫描全部 {num_layers} 个 transformer 层: {cand_layers}")
    else:
        # 显式给出了列表
        cand_layers = [int(x) for x in raw_layers_cfg]
        print(f"[INFO] 使用配置中的候选层: {cand_layers}")

    # 存每层的 [num_samples, hidden_dim]
    states_by_layer = {L: [] for L in cand_layers}

    # 3) 对每个 prompt 做一次 forward，拿“最后一个 token”的 hidden
    for prompt in tqdm(prompts, desc="Collect hidden states"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = out.hidden_states  # tuple: [emb, layer1, ..., layerN]
        # hidden_states[k].shape = (batch, seq_len, dim)

        for L in cand_layers:
            if L >= len(hidden_states):
                raise ValueError(
                    f"candidate layer {L} 超过了 hidden_states 数量 {len(hidden_states)}"
                )
            # 这里沿用原逻辑：取最后一个 token 的 hidden
            h_L = hidden_states[L][0, -1, :]  # batch=0, last token
            h_np = h_L.detach().cpu().float().numpy()
            states_by_layer[L].append(h_np)

    # 转成 numpy
    for L in cand_layers:
        states_by_layer[L] = np.stack(states_by_layer[L], axis=0)  # [N, D]
        print(f"[INFO] Layer {L}: states shape = {states_by_layer[L].shape}")

    # 4) 对每一层计算拒绝方向 + AUROC
    directions = {}
    layer_stats = []

    for L in cand_layers:
        H = states_by_layer[L]
        y = labels

        # 拆成拒绝 / 其他
        mask_ref = (y == 1)
        mask_oth = (y == 0)
        H_ref = H[mask_ref]
        H_oth = H[mask_oth]

        if len(H_ref) == 0 or len(H_oth) == 0:
            print(f"[WARN] Layer {L} 没有足够的 REFUSAL/OTHER 样本，跳过")
            continue

        H_ref_n = l2_normalize(H_ref)
        H_oth_n = l2_normalize(H_oth)

        mu_ref = H_ref_n.mean(axis=0)
        mu_oth = H_oth_n.mean(axis=0)

        d = mu_ref - mu_oth
        d = d / (np.linalg.norm(d) + 1e-8)

        directions[L] = d

        # 用该方向对所有样本打分，计算 AUROC
        H_all_n = l2_normalize(H)
        scores = (H_all_n @ d).astype(np.float32)  # [N]

        auroc = build_auroc(scores, y)
        layer_stats.append((L, auroc, int(mask_ref.sum()), int(mask_oth.sum())))
        print(f"[INFO] Layer {L}: AUROC={auroc:.4f}, REF={mask_ref.sum()}, OTH={mask_oth.sum()}")

    # 5) 保存扫描结果 CSV
    stats_path = dir_root / "text_direction_layer_scan_qwen2_5_vl_7b.csv"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("layer,auroc,num_refusal,num_other\n")
        for L, auroc, nR, nO in layer_stats:
            f.write(f"{L},{auroc:.6f},{nR},{nO}\n")
    print(f"[INFO] 层扫描结果写入: {stats_path}")

    # 6) 选出 AUROC 最高的若干层（例如 top-3）
    layer_stats_sorted = sorted(
        layer_stats,
        key=lambda x: (x[1] if not np.isnan(x[1]) else -1.0),
        reverse=True,
    )
    top_k = min(3, len(layer_stats_sorted))
    selected_layers = [layer_stats_sorted[i][0] for i in range(top_k)]
    print(f"[INFO] 选出的层 (按 AUROC 排名): {selected_layers}")

    if len(selected_layers) == 0:
        print("[WARN] 没有可用的层，无法保存方向与拒绝 token")
        return

    # 只保留选中的方向
    npz_path = dir_root / "text_directions_selected_qwen2_5_vl_7b.npz"
    np.savez(
        npz_path,
        selected_layers=np.array(selected_layers, dtype=np.int32),
        directions=np.stack([directions[L] for L in selected_layers], axis=0),
        layer_ids=np.array(selected_layers, dtype=np.int32),
    )
    print(f"[INFO] 选中层的方向写入: {npz_path}")

    # 7) 从 AUROC 最好的层抽拒绝 token
    best_layer = layer_stats_sorted[0][0]
    d_best = directions[best_layer].astype(np.float32)
    print(f"[INFO] 使用 layer {best_layer} 的方向提取拒绝 token")

    with torch.no_grad():
        emb = model.get_input_embeddings().weight  # [V, D]
        emb = emb.float()
        emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        d_t = torch.from_numpy(d_best).to(emb_norm.device)
        scores = emb_norm @ d_t  # [V]
        scores_np = scores.cpu().numpy()

    top_k_tokens = 200
    top_idx = np.argsort(scores_np)[-top_k_tokens:][::-1]
    tok_list = []
    for idx in top_idx:
        token_str = processor.tokenizer.convert_ids_to_tokens(int(idx))
        tok_list.append(
            {
                "token_id": int(idx),
                "token": token_str,
                "score": float(scores_np[idx]),
            }
        )

    refusal_tok_path = dir_root / "refusal_keywords_qwen2_5_vl_7b.json"
    with refusal_tok_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_layer": int(best_layer),
                "selected_layers": selected_layers,
                "refusal_tokens_topk": tok_list,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] 拒绝 token 写入: {refusal_tok_path}")


if __name__ == "__main__":
    main()
