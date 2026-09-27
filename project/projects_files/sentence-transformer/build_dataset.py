"""
Build a labelled multi-task receipt dataset from CORD-v2.

CORD groups the lines of a receipt into rows via `group_id`, so a row such as
group 3 below becomes one training sentence:

    menu.cnt    "1"
    menu.nm     "REAL GANACHE"      ->   "1 REAL GANACHE 16,500"
    menu.price  "16,500"

Two labels are derived per sentence, matching the heads the model declares:

  sentence class (4)   0 ITEM, 1 ITEM_OPTION, 2 SUBTOTAL, 3 TOTAL
  NER tags (5)         0 O, 1 B-NAME, 2 I-NAME, 3 B-PRICE, 4 I-PRICE

Usage:  python build_dataset.py
"""
from __future__ import annotations

import glob, json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

CACHE = "/Users/oliviajackson/Documents/portfolio/projects/receipt-data"
REPO = "naver-clova-ix/cord-v2"

FILES = {
    "train": ["data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
              "data/train-00001-of-00004-7dbbe248962764c5.parquet",
              "data/train-00002-of-00004-688fe1305a55e5cc.parquet",
              "data/train-00003-of-00004-2d0cd200555ed7fd.parquet"],
    "validation": ["data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"],
    "test": ["data/test-00000-of-00001-9c204eb3f4e11791.parquet"],
}

CLASS_NAMES = ["ITEM", "ITEM_OPTION", "SUBTOTAL", "TOTAL"]
NER_NAMES = ["O", "B-NAME", "I-NAME", "B-PRICE", "I-PRICE"]


def sentence_class(cats: list[str]) -> int | None:
    """Row category from the CORD hierarchy its lines belong to."""
    if any(c.startswith("menu.sub.") for c in cats):
        return 1
    if any(c.startswith("menu.") for c in cats):
        return 0
    if any(c.startswith("sub_total.") for c in cats):
        return 2
    if any(c.startswith("total.") for c in cats):
        return 3
    return None


def ner_type(cat: str) -> str | None:
    """Which entity, if any, a line's words belong to."""
    leaf = cat.split(".")[-1]
    if leaf == "nm":
        return "NAME"
    if "price" in leaf and leaf != "menuqty_cnt":
        return "PRICE"
    return None


def build_split(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        df = pd.read_parquet(p)
        for gt_raw in df["ground_truth"]:
            gt = json.loads(gt_raw)
            groups: dict[int, list] = {}
            for line in gt["valid_line"]:
                groups.setdefault(line["group_id"], []).append(line)

            for gid, lines in groups.items():
                cats = [l["category"] for l in lines]
                cls = sentence_class(cats)
                if cls is None:
                    continue
                words, tags = [], []
                for line in lines:
                    ent = ner_type(line["category"])
                    toks = [w["text"] for w in line["words"] if w["text"].strip()]
                    for j, tok in enumerate(toks):
                        words.append(tok)
                        if ent is None:
                            tags.append(0)
                        else:
                            tags.append(NER_NAMES.index(("B-" if j == 0 else "I-") + ent))
                if len(words) < 2:
                    continue
                rows.append({"words": words, "ner": tags, "cls": cls})
    return rows


def main() -> None:
    out = Path(CACHE) / "cord_multitask"
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split, files in FILES.items():
        local = [hf_hub_download(REPO, f, repo_type="dataset", cache_dir=CACHE) for f in files]
        rows = build_split(local)
        (out / f"{split}.json").write_text(json.dumps(rows))
        dist = {CLASS_NAMES[i]: sum(1 for r in rows if r["cls"] == i) for i in range(4)}
        summary[split] = (len(rows), dist)
        print(f"{split:<11} {len(rows):>6} sentences   {dist}")

    ner_counts = {n: 0 for n in NER_NAMES}
    for r in json.loads((out / "train.json").read_text()):
        for t in r["ner"]:
            ner_counts[NER_NAMES[t]] += 1
    print("\ntrain NER tag counts:", ner_counts)

    print("\nexamples:")
    for r in json.loads((out / "train.json").read_text())[:4]:
        pairs = " ".join(f"{w}/{NER_NAMES[t]}" for w, t in zip(r["words"], r["ner"]))
        print(f"   [{CLASS_NAMES[r['cls']]}] {pairs}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
