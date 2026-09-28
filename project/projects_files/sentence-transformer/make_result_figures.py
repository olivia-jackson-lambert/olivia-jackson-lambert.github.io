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
from figstyle import apply, tidy, INK, SLATE, RUST, DIM, GRID, MUTED, LIGHT  # noqa: E402

PANEL = dict(loc="left", fontsize=10.5, color=DIM, pad=10)   # small panel label, not a chart title
CLASS_NAMES = ["ITEM", "ITEM_OPTION", "SUBTOTAL", "TOTAL"]
NER_NAMES = ["O", "B-NAME", "I-NAME", "B-PRICE", "I-PRICE"]
TAG_COLOR = {"O": MUTED, "B-NAME": SLATE, "I-NAME": "#5d7390", "B-PRICE": RUST, "I-PRICE": "#b5645a"}
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
    H = {k: np.array([h[k] for h in hist]) for k in hist[0]}
    ep = H["epoch"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=False)
    for ax, task, label in ((a1, "cls", "Sentence classification"), (a2, "ner", "Entity tagging")):
        ax.plot(ep, H[f"train_{task}_loss"], color=SLATE, label="Train")
        ax.plot(ep, H[f"val_{task}_loss"], color=RUST, label="Validation")
        ax.set_title(label, **PANEL)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
        ax.set_xlim(1, ep.max()); ax.set_ylim(bottom=0)
        tidy(ax)
    a1.legend(loc="center right")
    best = int(np.argmin(H["val_cls_loss"])) + 1
    a1.axvline(best, color=MUTED, lw=1, ls=":")
    a1.annotate(f"validation minimum, epoch {best}", xy=(best, H["val_cls_loss"].min()),
                xytext=(best + 3, H["val_cls_loss"].max() * .55), fontsize=9, color=DIM,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=.8))
    fig.tight_layout(w_pad=3)
    save(fig, "training_curves.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    a1.plot(ep, H["train_cls_acc"], color=SLATE, label="Train accuracy")
    a1.plot(ep, H["val_cls_acc"], color=RUST, label="Validation accuracy")
    a1.plot(ep, H["val_cls_f1"], color=MUTED, lw=1.4, ls="--", label="Validation F1 (weighted)")
    a1.set_title("Sentence classification", **PANEL)
    a2.plot(ep, H["train_ner_acc"], color=SLATE, label="Train token accuracy")
    a2.plot(ep, H["val_ner_acc"], color=RUST, label="Validation token accuracy")
    a2.plot(ep, H["val_ner_f1"], color=MUTED, lw=1.4, ls="--", label="Validation F1 (macro)")
    a2.set_title("Entity tagging", **PANEL)
    for ax in (a1, a2):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score (0 to 1)")
        ax.set_xlim(1, ep.max())
        ax.legend(loc="lower right"); tidy(ax)
    fig.tight_layout(w_pad=3)
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
    fig = ax.figure
    r = fig.canvas.get_renderer()
    ax_w = ax.get_window_extent(r).width
    pad_px = 12 * fig.dpi / 72 * 0.32 * 2          # bbox pad on both sides, in pixels
    x = 0.0
    for i, (w, t) in enumerate(zip(words, tags)):
        col = TAG_COLOR[t]
        wrong = gold is not None and gold[i] != t
        t1 = ax.text(x, 0.66, w, fontsize=12, color=INK, transform=ax.transAxes,
                     ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.32", fc=col + "1f", ec=col,
                               lw=1.6 if wrong else 1.0, ls="--" if wrong else "-"))
        t2 = ax.text(x, 0.22, t, fontsize=9, color=col, transform=ax.transAxes,
                     ha="left", va="center", fontweight="medium")
        # advance by the wider of the word (plus its box padding) and its tag, measured in pixels
        wpx = max(t1.get_window_extent(r).width + pad_px, t2.get_window_extent(r).width)
        x += wpx / ax_w + 0.018


def example(name, words, gold_tags, gold_cls, tok, backbone, heads, dev):
    cls, probs, tags = predict(words, tok, backbone, heads, dev)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.0, 3.1), height_ratios=[1.6, 1.0])
    token_panel(a1, words, tags, gold_tags)
    n_wrong = sum(g != t for g, t in zip(gold_tags, tags))
    ok = "correct" if cls == gold_cls else f"wrong, true class {gold_cls}"
    note_txt = f"Sentence class: {cls} ({ok}).   Tags matching the reference: {len(tags) - n_wrong} of {len(tags)}"
    if n_wrong:
        note_txt += "   Dashed outline: tag differs from the reference."
    a1.text(0.0, -0.12, note_txt, transform=a1.transAxes, fontsize=9, color=DIM)
    order = np.argsort(probs)
    a2.barh([CLASS_NAMES[i] for i in order], probs[order],
            color=[RUST if CLASS_NAMES[i] == cls else LIGHT for i in order], height=.62)
    a2.set_xlim(0, 1); a2.set_xlabel("Predicted class probability")
    tidy(a2, grid_axis="x")
    fig.tight_layout(h_pad=1.6)
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

    fig = plt.figure(figsize=(12.6, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.2], wspace=0.04)
    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.axis("off")
    ax.set_title("Receipt image (CORD-v2 test set)", **PANEL)

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    fig.canvas.draw()                      # match the table's height to the photo's
    b, q = ax.get_position(), ax2.get_position()
    ax2.set_position([q.x0, b.y0, q.width, b.height])
    ax2.set_title("Model output, line by line", **PANEL)
    COLS = (0.00, 0.15, 0.55, 0.78)
    y = 0.98
    step = min(0.072, 0.86 / max(len(rows), 1))
    for x, h in zip(COLS, ("CLASS", "LINE", "NAME SPAN", "PRICE SPAN")):
        ax2.text(x, y, h, fontsize=8.5, color=MUTED, transform=ax2.transAxes, fontweight="medium")
    ax2.plot([0, 1], [y - step * 0.35] * 2, color=GRID, lw=1, transform=ax2.transAxes, clip_on=False)
    y -= step * 0.9
    for cls, line, name, price, right in rows:
        ax2.text(COLS[0], y, cls, fontsize=8.5, color=DIM, transform=ax2.transAxes, va="center")
        ax2.text(COLS[1], y, line if len(line) <= 40 else line[:39] + "…", fontsize=9,
                 color=INK, transform=ax2.transAxes, va="center")
        ax2.text(COLS[2], y, name if name else "none", fontsize=9,
                 color=SLATE if name else LIGHT, transform=ax2.transAxes, va="center")
        pr = price if len(price) <= 24 else price[:23] + "…"
        ax2.text(COLS[3], y, pr if price else "none", fontsize=9,
                 color=(RUST if not right else INK) if price else LIGHT,
                 transform=ax2.transAxes, va="center")
        if not right:
            ax2.add_patch(plt.Rectangle((-0.01, y - step * 0.42), 1.02, step * 0.84, transform=ax2.transAxes,
                                        fc="none", ec=RUST, lw=1.2, ls="--", clip_on=False))
        y -= step
    ax2.text(0.0, y - step * 0.2,
             "Spans are what the entity head tagged. Dashed box: row differs from the CORD reference labels.",
             fontsize=8.5, color=MUTED, transform=ax2.transAxes)
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
