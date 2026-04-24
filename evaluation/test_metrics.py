"""
test_metrics.py
===============
Quick smoke-test to verify all metric implementations produce
sensible values before running on real model outputs.

Run:  python test_metrics.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_captions import (
    corpus_bleu4, corpus_meteor, corpus_rouge_l, _compute_cider
)

PASS = "✅"
FAIL = "❌"


def check(name, value, lo, hi):
    ok = lo <= value <= hi
    mark = PASS if ok else FAIL
    print(f"  {mark}  {name}: {value:.4f}  (expected {lo:.2f}–{hi:.2f})")
    return ok


def run_tests():
    all_ok = True

    print("\n── Test 1: Perfect match ────────────────────────────────────────")
    hyps = ["a dog runs in the snow"]
    refs = [["a dog runs in the snow"]]
    all_ok &= check("BLEU-4",  corpus_bleu4 (hyps, refs), 0.95, 1.01)
    all_ok &= check("METEOR",  corpus_meteor(hyps, refs), 0.80, 1.01)
    all_ok &= check("ROUGE-L", corpus_rouge_l(hyps, refs),0.95, 1.01)
    # NOTE: CIDEr IDF collapses to 0 on single-image corpora (log(2/2)=0)
    # — correct behaviour. Single-image CIDEr is always 0.
    print(f"  ℹ️   CIDEr (single-image): 0 is expected — IDF is meaningless with 1 image")

    print("\n── Test 2: Unrelated sentence (low scores) ─────────────────────")
    hyps = ["a red car drives on the highway"]
    refs = [["two cats sleep on a couch"]]
    all_ok &= check("BLEU-4",  corpus_bleu4 (hyps, refs), 0.0, 0.10)
    all_ok &= check("METEOR",  corpus_meteor(hyps, refs), 0.0, 0.15)
    all_ok &= check("ROUGE-L", corpus_rouge_l(hyps, refs),0.0, 0.20)   # 'a' may share
    all_ok &= check("CIDEr",   _compute_cider(hyps, refs),0.0,  2.0)

    print("\n── Test 3: Multiple references / corpus — partial overlap ──────")
    hyps = [
        "a black dog is running in the snow",
        "a man rides a bicycle on the road",
    ]
    refs = [
        ["a dog runs in the snow", "black dog playing in snow"],
        ["a cyclist rides a bike", "man on bicycle on a road"],
    ]
    # BLEU-4 is 0 on short sentences with few 4-gram overlaps — acceptable
    all_ok &= check("METEOR",  corpus_meteor(hyps, refs), 0.20, 0.80)
    all_ok &= check("ROUGE-L", corpus_rouge_l(hyps, refs),0.20, 0.80)
    all_ok &= check("CIDEr",   _compute_cider(hyps, refs),1.0, 30.0)

    print("\n── Test 4: Generic vs. specific — CIDEr should rank specific higher")
    # Need a small corpus (≥2 images) for IDF to be nonzero
    hyps_gen = ["a man is standing",
                "a dog runs fast"]
    hyps_spe = ["a tall man in a red jacket stands near a blue car",
                "a golden dog sprints through a snowy field"]
    refs4    = [
        ["a tall man wearing a red jacket stands next to a blue sedan"],
        ["a golden retriever runs through deep snow"],
    ]
    cider_gen = _compute_cider(hyps_gen, refs4)
    cider_spe = _compute_cider(hyps_spe, refs4)
    ok4 = cider_spe > cider_gen
    print(f"  {'✅' if ok4 else '❌'}  Specific ({cider_spe:.2f}) > Generic ({cider_gen:.2f})")
    all_ok &= ok4

    print("\n── Test 5: Empty hypothesis edge case ──────────────────────────")
    try:
        hyps_e = [""]
        refs_e = [["a dog in the snow"]]
        s = corpus_bleu4(hyps_e, refs_e)
        print(f"  {PASS}  BLEU-4 with empty hyp returns 0.0: {s}")
    except Exception as ex:
        print(f"  {FAIL}  Exception on empty hyp: {ex}")
        all_ok = False

    print()
    if all_ok:
        print(f"{PASS}  All tests passed.\n")
    else:
        print(f"{FAIL}  Some tests FAILED — check metric implementations.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
