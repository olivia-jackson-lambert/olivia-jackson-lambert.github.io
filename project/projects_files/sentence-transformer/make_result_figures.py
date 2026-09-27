"""
Result figures for the receipt multi-task model, from the trained heads.

Every number and prediction here comes from the run produced by train_heads.py.
Usage:  python make_result_figures.py
"""
from __future__ import annotations

import io, json, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer

ART = Path("/Users/oliviajackson/Documents/portfolio/projects/receipt-data/artifacts")
DATA = Path("/Users/oliviajackson/Documents/portfolio/projects/receipt-data/cord_multitask")
CACHE = "/Users/oliviajackson/Documents/portfolio/projects/receipt-data"
OUT = Path("assets")
FONTS = Path("/Users/oliviajackson/Documents/portfolio/projects/high-energy-particle-classifier/assets/fonts")

SLATE, RUST, INK, GRID = "#3A506B", "#9E2A2B", "#1b1b1f", "#e6e6ec"
TITLE_PAD = 16
CLASS_NAMES = ["ITEM", "ITEM_OPTION", "SUBTOTAL", "TOTAL"]
NER_NAMES = ["O", "B-NAME", "I-NAME", "B-PRICE", "I-PRICE"]
TAG_COLOR = {"O": "#8a8f98", "B-NAME": SLATE, "I-NAME": "#6b8299", "B-PRICE": RUST, "I-PRICE": "#c07a75"}
MAXLEN = 32


def house_style():
    for f in ("Lora-Regular.ttf", "Lora-SemiBold.ttf"):
        if (FONTS / f).exists():
            fm.fontManager.addfont(str(FONTS / f))
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "font.family": "Lora" if (FONTS / "Lora-Regular.ttf").exists() else "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
    })


def tidy(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c9c9d2")
    ax.tick_params(colors=INK, labelsize=9)


class Heads(nn.Module):
    def __init__(self, dim=1024, n_cls=4, n_ner=5):
        super().__init__()
        self.cls_head = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, n_cls))
        self.ner_head = nn.Linear(dim, n_ner)
    def forward(self, h): return self.cls_head(h[:, 0]), self.ner_head(h)


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig); print("  wrote", name)


# ---------------------------------------------------------------- curves
def curves(hist):
    H = {k: np.array([h[k] for h in hist]) for k in hist[0]}
    ep = H["epoch"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, task, title in ((a1, "cls", "Classification Loss"), (a2, "ner", "NER Loss")):
        ax.plot(ep, H[f"train_{task}_loss"], color=SLATE, lw=2, label="Train")
        ax.plot(ep, H[f"val_{task}_loss"], color=RUST, lw=2, ls="--", label="Validation")
        ax.set_title(title, pad=TITLE_PAD, color=INK)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
        ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
        ax.legend(frameon=False); tidy(ax)
    best = int(np.argmin(H["val_cls_loss"])) + 1
    a1.axvline(best, color=INK, lw=1, ls=":", alpha=.6)
    a1.annotate(f"val minimum, epoch {best}", xy=(best, H["val_cls_loss"].min()),
                xytext=(best + 1.5, H["val_cls_loss"].max() * .75), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=.8))
    fig.tight_layout(w_pad=2.5)
    save(fig, "training_curves.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    a1.plot(ep, H["train_cls_acc"], color=SLATE, lw=2, label="Train accuracy")
    a1.plot(ep, H["val_cls_acc"], color=RUST, lw=2, ls="--", label="Validation accuracy")
    a1.plot(ep, H["val_cls_f1"], color="#c07a75", lw=1.4, label="Validation F1 (weighted)")
    a1.set_title("Classification Accuracy", pad=TITLE_PAD, color=INK)
    a2.plot(ep, H["train_ner_acc"], color=SLATE, lw=2, label="Train token accuracy")
    a2.plot(ep, H["val_ner_acc"], color=RUST, lw=2, ls="--", label="Validation token accuracy")
    a2.plot(ep, H["val_ner_f1"], color="#c07a75", lw=1.4, label="Validation F1 (macro)")
    a2.set_title("NER Accuracy", pad=TITLE_PAD, color=INK)
    for ax in (a1, a2):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9); tidy(ax)
    fig.tight_layout(w_pad=2.5)
    save(fig, "accuracy_curves.png")


# ---------------------------------------------------------------- predictions
def predict(words, tok, backbone, heads, dev):
    enc = tok([words], is_split_into_words=True, truncation=True,
              max_length=MAXLEN, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        h = backbone(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state
        lc, ln = heads(h.float())
    wid = enc.word_ids(0)
    tags, seen = [], set()
    pred = ln.argmax(-1)[0].cpu().tolist()
    for pos, w in enumerate(wid):
        if w is not None and w not in seen:
            seen.add(w); tags.append(NER_NAMES[pred[pos]])
    cls_probs = torch.softmax(lc[0], -1).cpu().numpy()
    return CLASS_NAMES[int(cls_probs.argmax())], cls_probs, tags[:len(words)]


def token_panel(ax, words, tags, gold=None):
    ax.axis("off")
    x = 0.01
    for i, (w, t) in enumerate(zip(words, tags)):
        col = TAG_COLOR[t]
        wrong = gold is not None and gold[i] != t
        ax.text(x, 0.62, w, fontsize=12, color=INK, transform=ax.transAxes,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.30", fc=col + "22", ec=col, lw=1.6 if wrong else 1.0,
                          ls="--" if wrong else "-"))
        ax.text(x, 0.26, t, fontsize=8.2, color=col, transform=ax.transAxes, ha="left", va="center")
        x += max(len(w), len(t)) * 0.0175 + 0.022


def example(name, words, gold_tags, gold_cls, title, tok, backbone, heads, dev):
    cls, probs, tags = predict(words, tok, backbone, heads, dev)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.0, 2.9), height_ratios=[2.0, 1.0])
    token_panel(a1, words, tags, gold_tags)
    a1.set_title(title, pad=TITLE_PAD, color=INK, loc="left")
    if gold_tags is not None and any(g != t for g, t in zip(gold_tags, tags)):
        a1.text(0.01, -0.06, "dashed outline: model disagrees with the reference tag",
                transform=a1.transAxes, fontsize=7.8, color="#8a8f98", style="italic")
    order = np.argsort(probs)[::-1]
    a2.barh([CLASS_NAMES[i] for i in order][::-1], probs[order][::-1],
            color=[RUST if CLASS_NAMES[i] == cls else "#c8ccd4" for i in order][::-1], height=.62)
    a2.set_xlim(0, 1); a2.set_xlabel("Predicted class probability")
    a2.grid(axis="x", color=GRID, lw=.8); a2.set_axisbelow(True); tidy(a2)
    ok = "correct" if cls == gold_cls else f"WRONG, true class {gold_cls}"
    a2.text(1.0, -0.62, f"sentence class: {cls} ({ok})", transform=a2.transAxes,
            ha="right", fontsize=9, color=INK if cls == gold_cls else RUST)
    fig.tight_layout(h_pad=1.4)
    save(fig, name)


def complete_receipt(tok, backbone, heads, dev):
    p = glob.glob(f"{CACHE}/**/test-*.parquet", recursive=True)[0]
    df = pd.read_parquet(p)
    # choose a receipt with a decent number of line groups, so the table is full
    best_i, best_n = 0, 0
    for i in range(len(df)):
        g = {l["group_id"] for l in json.loads(df.iloc[i]["ground_truth"])["valid_line"]}
        if 9 <= len(g) <= 15 and len(g) > best_n:
            best_i, best_n = i, len(g)
    row = df.iloc[best_i]
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    gt = json.loads(row["ground_truth"])

    groups = {}
    for line in gt["valid_line"]:
        groups.setdefault(line["group_id"], []).append(line)

    rows = []
    for gid in sorted(groups):
        words = [w["text"] for l in groups[gid] for w in l["words"] if w["text"].strip()]
        if len(words) < 2:
            continue
        cls, _, tags = predict(words, tok, backbone, heads, dev)
        name = " ".join(w for w, t in zip(words, tags) if t.endswith("NAME"))
        price = " ".join(w for w, t in zip(words, tags) if t.endswith("PRICE"))
        rows.append((cls, " ".join(words), name, price))

    fig = plt.figure(figsize=(12.4, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.05], wspace=0.04)
    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.axis("off")
    ax.set_title("Receipt image", fontsize=11, pad=8, color=INK)

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.set_title("Model output, line by line", fontsize=11, pad=8, color=INK, loc="left")
    COLS = (0.00, 0.16, 0.56, 0.82)
    y = 0.95
    step = min(0.075, 0.88 / max(len(rows), 1))
    for x, h in zip(COLS, ("CLASS", "LINE", "NAME", "PRICE")):
        ax2.text(x, y, h, fontsize=8, color="#8a8f98", transform=ax2.transAxes)
    ax2.plot([0, 1], [y - step * 0.32] * 2, color=GRID, lw=1, transform=ax2.transAxes, clip_on=False)
    y -= step * 0.85
    for cls, line, name, price in rows:
        ax2.text(COLS[0], y, cls, fontsize=8.0, color=RUST if cls in ("TOTAL", "SUBTOTAL") else SLATE,
                 transform=ax2.transAxes, va="center")
        ax2.text(COLS[1], y, line[:38], fontsize=8.6, color=INK, transform=ax2.transAxes, va="center")
        ax2.text(COLS[2], y, name[:24] if name else "\u2013", fontsize=8.6,
                 color=SLATE if name else "#b9bec7", transform=ax2.transAxes, va="center")
        ax2.text(COLS[3], y, price[:18] if price else "\u2013", fontsize=8.6,
                 color=RUST if price else "#b9bec7", transform=ax2.transAxes, va="center")
        y -= step
    ax2.text(0.0, y - step * 0.3, "NAME and PRICE columns are spans the NER head tagged, not rule-based parses.",
             fontsize=7.8, color="#8a8f98", transform=ax2.transAxes, style="italic")
    fig.suptitle("End-to-End Receipt Processing", fontsize=13, fontweight="semibold", color=INK, y=0.99)
    save(fig, "complete_receipt_example.png")


def main():
    house_style(); OUT.mkdir(exist_ok=True)
    hist = json.loads((ART / "history.json").read_text())
    curves(hist)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    backbone = AutoModel.from_pretrained("BAAI/bge-m3").to(dev).eval()
    heads = Heads().to(dev); heads.load_state_dict(torch.load(ART / "heads.pt")); heads.eval()

    test = json.loads((DATA / "test.json").read_text())
    def alpha_name(r):
        """Name tokens that are actually words, so the example reads sensibly."""
        toks = [w for w, t in zip(r["words"], r["ner"]) if t in (1, 2)]
        return toks and all(any(c.isalpha() for c in w) for w in toks) and len("".join(toks)) >= 6

    def pick(cls_idx, min_words=4, need_name=True):
        cands = [r for r in test if r["cls"] == cls_idx and len(r["words"]) >= min_words
                 and (alpha_name(r) if need_name else True)]
        return cands[0] if cands else next(r for r in test if r["cls"] == cls_idx)

    item = pick(0)
    example("example_product_classification.png", item["words"],
            [NER_NAMES[t] for t in item["ner"]], CLASS_NAMES[item["cls"]],
            "Product line: name and price extracted from one item row", tok, backbone, heads, dev)

    item2 = [r for r in test if r["cls"] == 0 and sum(1 for t in r["ner"] if t >= 3) >= 2
             and alpha_name(r)][0]
    example("example_price_extraction.png", item2["words"],
            [NER_NAMES[t] for t in item2["ner"]], CLASS_NAMES[item2["cls"]],
            "Price extraction: multi-token price spans tagged B-PRICE then I-PRICE", tok, backbone, heads, dev)

    tot = pick(3, need_name=False)
    example("example_total_line.png", tot["words"],
            [NER_NAMES[t] for t in tot["ner"]], CLASS_NAMES[tot["cls"]],
            "Total line: same heads, different sentence class", tok, backbone, heads, dev)

    complete_receipt(tok, backbone, heads, dev)


if __name__ == "__main__":
    main()
