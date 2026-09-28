"""
Result figures for the receipt multi-task model, from the trained heads.

Every number and prediction here comes from the run produced by train_heads.py.
Usage:  python make_result_figures.py
"""
from __future__ import annotations

import io, json, glob, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import (apply, tidy, title_case, INK, SLATE, RUST, DIM, GRID, MUTED, LIGHT,  # noqa: E402
                      SLATE_TINT, RUST_TINT, PAPER)

PANEL = dict(loc="left", fontsize=10.5, color=DIM, pad=10)   # small panel label, not a chart title
CLASS_NAMES = ["ITEM", "ITEM_OPTION", "SUBTOTAL", "TOTAL"]
NER_NAMES = ["O", "B-NAME", "I-NAME", "B-PRICE", "I-PRICE"]
TAG_COLOR = {"O": MUTED, "B-NAME": SLATE, "I-NAME": SLATE, "B-PRICE": RUST, "I-PRICE": RUST}
TAG_FILL = {"O": PAPER, "B-NAME": SLATE_TINT, "I-NAME": SLATE_TINT, "B-PRICE": RUST_TINT, "I-PRICE": RUST_TINT}
TOKENS_PER_ROW = 4
MAXLEN = 32


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
    """Two panels stacked vertically with a shared epoch axis, for a narrow column."""
    H = {k: np.array([h[k] for h in hist]) for k in hist[0]}
    ep = H["epoch"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
    for ax, task, label in ((a1, "cls", "Sentence Classification"), (a2, "ner", "Entity Tagging")):
        ax.plot(ep, H[f"train_{task}_loss"], color=SLATE, label="Train")
        ax.plot(ep, H[f"val_{task}_loss"], color=RUST, label="Validation")
        ax.set_title(label, **PANEL)
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_xlim(1, ep.max()); ax.set_ylim(bottom=0)
        tidy(ax)
    a2.set_xlabel("Epoch")
    a1.legend(loc="center right")
    best = int(np.argmin(H["val_cls_loss"])) + 1
    a1.axvline(best, color=MUTED, lw=1, ls=":")
    a1.annotate(f"Validation Minimum, Epoch {best}", xy=(best, H["val_cls_loss"].min()),
                xytext=(best + 3, H["val_cls_loss"].max() * .45), fontsize=9.5, color=DIM,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=.8))
    a1.tick_params(labelbottom=True)
    fig.tight_layout(h_pad=2.5)
    save(fig, "training_curves.png")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
    a1.plot(ep, H["train_cls_acc"], color=SLATE, label="Train Accuracy")
    a1.plot(ep, H["val_cls_acc"], color=RUST, label="Validation Accuracy")
    a1.plot(ep, H["val_cls_f1"], color=MUTED, lw=1.4, ls="--", label="Validation F1 (Weighted)")
    a1.set_title("Sentence Classification", **PANEL)
    a2.plot(ep, H["train_ner_acc"], color=SLATE, label="Train Token Accuracy")
    a2.plot(ep, H["val_ner_acc"], color=RUST, label="Validation Token Accuracy")
    a2.plot(ep, H["val_ner_f1"], color=MUTED, lw=1.4, ls="--", label="Validation F1 (Macro)")
    a2.set_title("Entity Tagging", **PANEL)
    for ax in (a1, a2):
        ax.set_ylabel("Score (0 to 1)")
        ax.set_xlim(1, ep.max())
        ax.legend(loc="lower right"); tidy(ax)
    a2.set_xlabel("Epoch")
    a1.tick_params(labelbottom=True)
    fig.tight_layout(h_pad=2.5)
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
    """Word boxes with their predicted tag beneath, wrapped TOKENS_PER_ROW to a line."""
    ax.axis("off")
    fig = ax.figure
    r = fig.canvas.get_renderer()
    ax_w = ax.get_window_extent(r).width
    pad_px = 12 * fig.dpi / 72 * 0.32 * 2          # bbox pad on both sides, in pixels
    n_rows = int(np.ceil(len(words) / TOKENS_PER_ROW))
    row_h = 1.0 / n_rows
    for i, (w, t) in enumerate(zip(words, tags)):
        row, col = divmod(i, TOKENS_PER_ROW)
        if col == 0:
            x = 0.012
        y_word = 1 - row_h * (row + 0.30)
        y_tag = 1 - row_h * (row + 0.72)
        wrong = gold is not None and gold[i] != t
        t1 = ax.text(x, y_word, w, fontsize=12, color=INK, transform=ax.transAxes,
                     ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.32", fc=TAG_FILL[t], ec=TAG_COLOR[t],
                               lw=1.6 if wrong else 1.0, ls="--" if wrong else "-"))
        t2 = ax.text(x, y_tag, t, fontsize=9.5, color=TAG_COLOR[t], transform=ax.transAxes,
                     ha="left", va="center", fontweight="semibold")
        wpx = max(t1.get_window_extent(r).width + pad_px, t2.get_window_extent(r).width)
        x += wpx / ax_w + 0.035


def example(name, words, gold_tags, gold_cls, tok, backbone, heads, dev):
    cls, probs, tags = predict(words, tok, backbone, heads, dev)
    n_rows = int(np.ceil(len(words) / TOKENS_PER_ROW))
    fig, (a1, a0, a2) = plt.subplots(3, 1, figsize=(6.4, 1.0 * n_rows + 2.3),
                                     height_ratios=[1.0 * n_rows, 0.7, 1.2])
    token_panel(a1, words, tags, gold_tags)
    n_wrong = sum(g != t for g, t in zip(gold_tags, tags))
    ok = "Correct" if cls == gold_cls else f"Wrong, True Class {gold_cls}"
    notes = [f"Sentence Class: {cls} ({ok})",
             title_case(f"Tags matching the reference: {len(tags) - n_wrong} of {len(tags)}")]
    if n_wrong:
        notes.append(title_case("Dashed outline: tag differs from the reference"))
    a0.axis("off")
    a0.text(0.012, 0.5, "\n".join(notes), transform=a0.transAxes, fontsize=9.5, color=DIM,
            va="center", linespacing=1.5)
    order = np.argsort(probs)
    a2.barh([CLASS_NAMES[i] for i in order], probs[order],
            color=[RUST if CLASS_NAMES[i] == cls else LIGHT for i in order], height=.62)
    a2.set_xlim(0, 1); a2.set_xlabel("Predicted Class Probability")
    tidy(a2, grid_axis="x")
    fig.tight_layout(h_pad=0.8)
    save(fig, name)
    print(f"    {name}: class {cls} (gold {gold_cls}), {len(tags)-n_wrong}/{len(tags)} tags match")


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

    # Reference labels come from the same rules as build_dataset.py.
    sys.path.insert(0, str(HERE))
    from build_dataset import sentence_class, ner_type

    rows, n_ok = [], 0
    for gid in sorted(groups):
        words, gold = [], []
        for l in groups[gid]:
            ent = ner_type(l["category"])
            toks = [w["text"] for w in l["words"] if w["text"].strip()]
            for j, t in enumerate(toks):
                words.append(t)
                gold.append("O" if ent is None else ("B-" if j == 0 else "I-") + ent)
        if len(words) < 2:
            continue
        gcls = sentence_class([l["category"] for l in groups[gid]])
        cls, _, tags = predict(words, tok, backbone, heads, dev)
        right = (gcls is not None and cls == CLASS_NAMES[gcls] and tags == gold[:len(tags)])
        n_ok += right
        print(f"    {'ok ' if right else 'ERR'} {cls:<11} gold {CLASS_NAMES[gcls] if gcls is not None else None:<11} "
              f"{' '.join(words)}  pred={tags}  gold={gold}")
        name = " ".join(w for w, t in zip(words, tags) if t.endswith("NAME"))
        price = " ".join(w for w, t in zip(words, tags) if t.endswith("PRICE"))
        rows.append((cls, " ".join(words), name, price, right))
    print(f"    complete receipt: {n_ok} of {len(rows)} rows fully correct")

    fig = plt.figure(figsize=(8.4, 11.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.12)
    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.axis("off")
    ax.set_title("Receipt Image (CORD-v2 Test Set)", **PANEL)

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.set_title("Model Output, Line by Line", **PANEL)
    COLS = (0.00, 0.17, 0.60, 0.82)
    y = 0.98
    step = min(0.068, 0.86 / max(len(rows), 1))
    for x, h in zip(COLS, ("Class", "Line", "Name Span", "Price Span")):
        ax2.text(x, y, h, fontsize=9.5, color=MUTED, transform=ax2.transAxes, fontweight="semibold")
    ax2.plot([0, 1], [y - step * 0.4] * 2, color=GRID, lw=1, transform=ax2.transAxes, clip_on=False)
    y -= step * 0.95
    for cls, line, name, price, right in rows:
        if not right:
            ax2.add_patch(plt.Rectangle((-0.01, y - step * 0.42), 1.02, step * 0.84, transform=ax2.transAxes,
                                        fc=RUST_TINT, ec=RUST, lw=1.0, ls="--", clip_on=False, zorder=0))
        ax2.text(COLS[0], y, cls, fontsize=8.8, color=DIM, transform=ax2.transAxes, va="center")
        ax2.text(COLS[1], y, line if len(line) <= 34 else line[:33] + "\u2026", fontsize=9,
                 color=INK, transform=ax2.transAxes, va="center")
        ax2.text(COLS[2], y, name if name else "None", fontsize=9,
                 color=SLATE if name else MUTED, transform=ax2.transAxes, va="center")
        pr = price if len(price) <= 15 else price[:14] + "\u2026"
        ax2.text(COLS[3], y, pr if price else "None", fontsize=9,
                 color=(RUST if not right else INK) if price else MUTED,
                 transform=ax2.transAxes, va="center")
        y -= step
    ax2.text(0.0, y - step * 0.15,
             "Spans Are What the Entity Head Tagged\n"
             "Shaded Rows: Tags Differ from the CORD Reference Labels",
             fontsize=9, color=MUTED, transform=ax2.transAxes, va="top", linespacing=1.5)
    save(fig, "complete_receipt_example.png")


def main():
    apply(); OUT.mkdir(exist_ok=True)
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

    def run(name, r):
        example(name, r["words"], [NER_NAMES[t] for t in r["ner"]], CLASS_NAMES[r["cls"]],
                tok, backbone, heads, dev)

    run("example_product_classification.png", pick(0))
    # a line whose reference contains a real multi-token price span (B-PRICE followed by I-PRICE)
    run("example_price_extraction.png",
        [r for r in test if r["cls"] == 0 and 4 in r["ner"] and alpha_name(r)][0])
    run("example_total_line.png", pick(3, need_name=False))

    complete_receipt(tok, backbone, heads, dev)


if __name__ == "__main__":
    main()
