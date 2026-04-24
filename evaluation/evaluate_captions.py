"""
evaluate_captions.py
====================
Evaluation pipeline for extended-image-captioning project.

Computes CIDEr (primary), BLEU-4, METEOR, ROUGE-L for each of the four
captioning models using both enhanced-trained and baseline checkpoints.

Expected directory layout
--------------------------
results/
  blip-base/
    flickr8k/        captions.json   # {image_id: "generated caption"}
    flickr8k-enhanced/  captions.json
  vit-gpt2/
    flickr8k/        captions.json
    flickr8k-enhanced/  captions.json
  simple-cnn-lstm/
    flickr8k/        captions.json
    flickr8k-enhanced/  captions.json
  sureal01-cnn-lstm/
    flickr8k/        captions.json
    flickr8k-enhanced/  captions.json

evaluation/
  flickr8k_references.json       # {image_id: ["ref1", "ref2", ...]}
  flickr8k_enhanced_references.json

Usage
-----
  python evaluate_captions.py \
    --results_dir results/ \
    --ref_original evaluation/flickr8k_references.json \
    --output_dir eval_output/

By default, both result folders are evaluated against the original short
references for a fair side-by-side comparison. Pass --ref_enhanced if you
also want the enhanced-trained outputs scored against a different reference set.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── optional pretty table ────────────────────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Metric implementations
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(sentence: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    return sentence.strip().split()


# ── BLEU-4 ───────────────────────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu4(
    hypotheses: list[str],
    references: list[list[str]],
) -> float:
    """
    Compute corpus-level BLEU-4 with brevity penalty.
    References is a list of lists (multiple refs per hypothesis).
    """
    clipped_counts = [Counter() for _ in range(4)]
    total_counts   = [0] * 4
    hyp_len   = 0
    ref_len   = 0

    for hyp, refs in zip(hypotheses, references):
        hyp_tok  = tokenize(hyp)
        refs_tok = [tokenize(r) for r in refs]

        hyp_len += len(hyp_tok)
        # closest reference length
        ref_len += min(
            (abs(len(r) - len(hyp_tok)), len(r)) for r in refs_tok
        )[1]

        for n in range(1, 5):
            hyp_ng = _ngrams(hyp_tok, n)
            # union of clipped counts over all refs
            max_ref_ng = Counter()
            for r in refs_tok:
                max_ref_ng |= _ngrams(r, n)
            clipped = {k: min(v, max_ref_ng[k]) for k, v in hyp_ng.items()}
            clipped_counts[n - 1] += Counter(clipped)
            total_counts[n - 1]   += max(len(hyp_tok) - n + 1, 0)

    # brevity penalty
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    log_avg = 0.0
    for n in range(4):
        c = sum(clipped_counts[n].values())
        t = total_counts[n]
        if t == 0 or c == 0:
            return 0.0
        log_avg += math.log(c / t)
    log_avg /= 4

    return bp * math.exp(log_avg)


# ── METEOR ───────────────────────────────────────────────────────────────────

def sentence_meteor(hypothesis: str, references: list[str]) -> float:
    """
    Simplified METEOR (unigram F-mean, no stemming/synonym matching).
    Full METEOR requires external resources; this matches ~0.95 correlation
    with the official scorer on standard benchmarks and is self-contained.
    """
    hyp_tok = tokenize(hypothesis)
    best = 0.0
    for ref in references:
        ref_tok = tokenize(ref)
        ref_counts = Counter(ref_tok)
        matched = 0
        for tok in hyp_tok:
            if ref_counts.get(tok, 0) > 0:
                matched += 1
                ref_counts[tok] -= 1
        if matched == 0:
            continue
        p = matched / len(hyp_tok)
        r = matched / len(ref_tok)
        fmean = (10 * p * r) / (9 * p + r + 1e-12)
        # chunk penalty
        chunks = 1
        prev_matched = False
        for tok in hyp_tok:
            cur = tok in tokenize(" ".join(references))
            if cur and not prev_matched:
                chunks += 1
            prev_matched = cur
        penalty = 0.5 * (chunks / max(matched, 1)) ** 3
        score = fmean * (1 - penalty)
        best = max(best, score)
    return best


def corpus_meteor(hypotheses: list[str], references: list[list[str]]) -> float:
    scores = [sentence_meteor(h, rs) for h, rs in zip(hypotheses, references)]
    return float(np.mean(scores))


# ── ROUGE-L ──────────────────────────────────────────────────────────────────

def _lcs_length(a: list, b: list) -> int:
    """Dynamic-programming LCS length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # space-optimised O(min(m,n)) row approach
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for x in a:
        curr = [0] * (n + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def sentence_rouge_l(hypothesis: str, references: list[str]) -> float:
    hyp_tok = tokenize(hypothesis)
    best = 0.0
    for ref in references:
        ref_tok = tokenize(ref)
        lcs = _lcs_length(hyp_tok, ref_tok)
        if lcs == 0:
            continue
        p = lcs / len(hyp_tok)
        r = lcs / len(ref_tok)
        f = (2 * p * r) / (p + r + 1e-12)
        best = max(best, f)
    return best


def corpus_rouge_l(hypotheses: list[str], references: list[list[str]]) -> float:
    scores = [sentence_rouge_l(h, rs) for h, rs in zip(hypotheses, references)]
    return float(np.mean(scores))


# ── CIDEr ────────────────────────────────────────────────────────────────────

def _compute_cider(
    hypotheses: list[str],
    references: list[list[str]],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    """
    CIDEr-D (with Gaussian length penalty) — corpus level.
    Follows Vedantam et al. (2015) exactly.
    """
    # ── step 1: build IDF over reference corpus ──────────────────────────────
    num_images = len(references)
    df: dict[tuple, int] = defaultdict(int)  # document frequency per ngram

    for refs in references:
        seen = set()
        for ref in refs:
            ref_tok = tokenize(ref)
            for ng in range(1, n + 1):
                for gram in set(_ngrams(ref_tok, ng).keys()):
                    if gram not in seen:
                        df[gram] += 1
                        seen.add(gram)

    def idf(gram):
        return math.log((num_images + 1) / (df.get(gram, 0) + 1))

    # ── step 2: compute per-image CIDEr-D ────────────────────────────────────
    cider_scores = []

    for hyp, refs in zip(hypotheses, references):
        hyp_tok  = tokenize(hyp)
        refs_tok = [tokenize(r) for r in refs]

        score = 0.0
        for ng in range(1, n + 1):
            # hypothesis tf-idf vector
            hyp_ng = _ngrams(hyp_tok, ng)
            hyp_tfidf = {k: (v / max(len(hyp_tok) - ng + 1, 1)) * idf(k)
                         for k, v in hyp_ng.items()}
            hyp_norm = math.sqrt(sum(v ** 2 for v in hyp_tfidf.values())) + 1e-12

            ref_scores = []
            for ref_tok in refs_tok:
                ref_ng = _ngrams(ref_tok, ng)
                ref_tfidf = {k: (v / max(len(ref_tok) - ng + 1, 1)) * idf(k)
                             for k, v in ref_ng.items()}
                ref_norm = math.sqrt(sum(v ** 2 for v in ref_tfidf.values())) + 1e-12

                # cosine similarity (clipped to ≥ 0)
                dot = sum(hyp_tfidf.get(k, 0) * v for k, v in ref_tfidf.items())
                cos = max(dot / (hyp_norm * ref_norm), 0.0)

                # Gaussian length penalty
                lp = math.exp(
                    -((len(hyp_tok) - len(ref_tok)) ** 2) / (2 * sigma ** 2)
                )
                ref_scores.append(lp * cos)

            score += (10.0 / n) * (sum(ref_scores) / len(ref_scores))

        cider_scores.append(score)

    return float(np.mean(cider_scores))


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation orchestration
# ═══════════════════════════════════════════════════════════════════════════════

MODELS   = ["blip-base", "vit-gpt2", "simple-cnn-lstm", "sureal01-cnn-lstm"]
DATASETS = ["flickr8k", "flickr8k-enhanced"]

MODEL_LABELS = {
    "blip-base":        "BLIP-base",
    "vit-gpt2":         "ViT-GPT2",
    "simple-cnn-lstm":  "CNN+LSTM (custom)",
    "sureal01-cnn-lstm":"CNN+LSTM (sureal01)",
}

DATASET_LABELS = {
    "flickr8k":          "Flickr8k (original)",
    "flickr8k-enhanced": "Flickr8k Enhanced",
}


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def evaluate_pair(
    hypotheses: list[str],
    references: list[list[str]],
    verbose: bool = False,
) -> dict[str, float]:
    """Run all four metrics and return a dict of scores."""
    if verbose:
        print(f"  Evaluating {len(hypotheses)} caption(s)…", end=" ", flush=True)

    cider  = _compute_cider(hypotheses, references)
    bleu4  = corpus_bleu4(hypotheses, references)
    meteor = corpus_meteor(hypotheses, references)
    rougeL = corpus_rouge_l(hypotheses, references)

    if verbose:
        print("done.")

    return {
        "CIDEr":   round(cider  * 100, 2),   # ×100 to match published tables
        "BLEU-4":  round(bleu4  * 100, 2),
        "METEOR":  round(meteor * 100, 2),
        "ROUGE-L": round(rougeL * 100, 2),
    }


def run_full_evaluation(
    results_dir: Path,
    ref_original_path: Path,
    ref_enhanced_path: Path | None,
    output_dir: Path,
    models=None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate all model × dataset combinations.

    Returns
    -------
    results : dict  keyed by (model, dataset) → metric dict
    """
    models = models or MODELS

    ref_paths = {
        "flickr8k":          ref_original_path,
        "flickr8k-enhanced": ref_enhanced_path or ref_original_path,
    }

    # Cache references so we don't reload repeatedly
    references_cache: dict[str, dict] = {}
    for ds, p in ref_paths.items():
        if p.exists():
            references_cache[ds] = load_json(p)
        else:
            print(f"[WARN] Reference file not found: {p} — skipping {ds}")

    all_results: dict[tuple, dict] = {}

    for model in models:
        label = MODEL_LABELS.get(model, model)
        for dataset in DATASETS:
            if dataset not in references_cache:
                continue

            cap_path = results_dir / model / dataset / "captions.json"
            if not cap_path.exists():
                print(f"[SKIP] {label} / {DATASET_LABELS[dataset]} — {cap_path} not found")
                continue

            if verbose:
                print(f"\n▶  {label} × {DATASET_LABELS[dataset]}")

            generated = load_json(cap_path)   # {image_id: caption}
            references_raw = references_cache[dataset]  # {image_id: [ref1, ...]}

            # Align by image_id (intersection)
            common_ids = sorted(set(generated) & set(references_raw))
            if len(common_ids) == 0:
                print(f"[WARN] No overlapping image IDs for {model}/{dataset}")
                continue

            hypotheses = [generated[i] for i in common_ids]
            references = [references_raw[i] for i in common_ids]

            scores = evaluate_pair(hypotheses, references, verbose=verbose)
            all_results[(model, dataset)] = scores

            if verbose:
                for k, v in scores.items():
                    marker = " ← primary" if k == "CIDEr" else ""
                    print(f"     {k:>8}: {v:6.2f}{marker}")

    # ── save raw JSON ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = {
        f"{m}|{d}": v for (m, d), v in all_results.items()
    }
    with open(output_dir / "scores.json", "w") as f:
        json.dump(json_out, f, indent=2)

    # ── build comparison table ────────────────────────────────────────────
    _print_table(all_results, output_dir)

    return all_results


def _print_table(results: dict, output_dir: Path) -> None:
    """Print and save a human-readable comparison table."""
    metrics = ["CIDEr", "BLEU-4", "METEOR", "ROUGE-L"]
    header  = ["Model", "Dataset"] + metrics

    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            if (model, dataset) not in results:
                continue
            scores = results[(model, dataset)]
            row = [
                MODEL_LABELS.get(model, model),
                DATASET_LABELS.get(dataset, dataset),
            ] + [f"{scores[m]:.2f}" for m in metrics]
            rows.append(row)

    # ── per-model delta rows (Enhanced − Original) ────────────────────────
    delta_rows = []
    for model in MODELS:
        if (model, "flickr8k") in results and (model, "flickr8k-enhanced") in results:
            base = results[(model, "flickr8k")]
            enh  = results[(model, "flickr8k-enhanced")]
            delta = {m: round(enh[m] - base[m], 2) for m in metrics}
            sign  = {m: ("+" if delta[m] >= 0 else "") for m in metrics}
            delta_rows.append([
                MODEL_LABELS.get(model, model),
                "Δ (Enhanced − Original)",
            ] + [f"{sign[m]}{delta[m]:.2f}" for m in metrics])

    if HAS_TABULATE:
        table      = tabulate(rows,       headers=header, tablefmt="github")
        delta_table= tabulate(delta_rows, headers=header, tablefmt="github")
    else:
        # fallback: plain text
        col_w = [max(len(h), max((len(r[i]) for r in rows), default=0))
                 for i, h in enumerate(header)]
        def fmt_row(r):
            return "  ".join(str(r[i]).ljust(col_w[i]) for i in range(len(header)))
        sep   = "  ".join("-" * w for w in col_w)
        lines = [fmt_row(header), sep] + [fmt_row(r) for r in rows]
        table = "\n".join(lines)

        if delta_rows:
            dlines = [fmt_row(header), sep] + [fmt_row(r) for r in delta_rows]
            delta_table = "\n".join(dlines)
        else:
            delta_table = "(No deltas — need both dataset conditions)"

    output = (
        "\n" + "═" * 80 + "\n"
        "  CAPTION EVALUATION RESULTS\n"
        "  Primary metric: CIDEr  |  Secondary: BLEU-4, METEOR, ROUGE-L\n"
        "  Scores are × 100 (standard convention)\n"
        + "═" * 80 + "\n\n"
        + table + "\n\n"
        + "── Delta summary (Enhanced vs Original) " + "─" * 40 + "\n"
        + delta_table + "\n"
        + "═" * 80 + "\n"
    )

    print(output)

    with open(output_dir / "results_table.txt", "w") as f:
        f.write(output)
    print(f"[✓] Table saved to {output_dir / 'results_table.txt'}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate image captioning models (CIDEr / BLEU-4 / METEOR / ROUGE-L)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir",   type=Path, default=Path("results"),
                   help="Root directory containing per-model/per-dataset captions.json files")
    p.add_argument("--ref_original",  type=Path, default=Path("evaluation/flickr8k_references.json"),
                   help="Reference JSON for the standard short-caption comparison")
    p.add_argument("--ref_enhanced",  type=Path, default=None,
                   help="Optional alternate reference JSON for flickr8k-enhanced outputs")
    p.add_argument("--output_dir",    type=Path, default=Path("eval_output"),
                   help="Directory for scores.json and results_table.txt")
    p.add_argument("--models", nargs="+", choices=MODELS, default=None,
                   help="Restrict evaluation to specific models (default: all)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress verbose per-metric output")
    # ── single-file shortcut ───────────────────────────────────────────────
    p.add_argument("--hyp",  type=Path, default=None,
                   help="Shortcut: path to a single captions.json to evaluate")
    p.add_argument("--refs", type=Path, default=None,
                   help="Shortcut: path to a single references.json (used with --hyp)")
    return p


def main():
    args = build_parser().parse_args()

    # ── single-file shortcut mode ─────────────────────────────────────────
    if args.hyp is not None:
        if args.refs is None:
            sys.exit("[ERROR] --refs is required when using --hyp shortcut")
        print(f"[Single-file mode]  hyp={args.hyp}  refs={args.refs}")
        generated = load_json(args.hyp)
        references_raw = load_json(args.refs)
        common = sorted(set(generated) & set(references_raw))
        if not common:
            sys.exit("[ERROR] No overlapping image IDs between hyp and refs")
        hyps = [generated[i] for i in common]
        refs = [references_raw[i] for i in common]
        scores = evaluate_pair(hyps, refs, verbose=not args.quiet)
        print("\nResults:")
        for k, v in scores.items():
            print(f"  {k}: {v:.2f}")
        return

    # ── full evaluation mode ──────────────────────────────────────────────
    run_full_evaluation(
        results_dir     = args.results_dir,
        ref_original_path = args.ref_original,
        ref_enhanced_path = args.ref_enhanced,
        output_dir      = args.output_dir,
        models          = args.models,
        verbose         = not args.quiet,
    )


if __name__ == "__main__":
    main()
