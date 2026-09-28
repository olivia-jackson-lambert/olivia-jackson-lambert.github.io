"""
Figure generation for the multi-task sentence transformer (receipt processing) write-up.

Two kinds of figure are produced, and the distinction is deliberate:

  SCHEMATICS  - diagrams of structure and data flow. These depend only on the
                code in olivia-jackson-lambert/sentence-project (model class,
                head dimensions, BGE-M3 config) and on the OCR functions defined
                in the notebook. No measured quantity is drawn.

  REAL DATA   - figures computed from actual BAAI/bge-m3 inference run locally by
                this script. The embeddings reproduce the notebook's recorded
                output exactly (nearest neighbours of sentence 1 score 0.747 and
                0.556, matching the saved cell output in
                notebooks/sentence_transformer.ipynb).

Result figures (training curves, example predictions, the complete receipt) are
made by make_result_figures.py from the heads trained by train_heads.py.

All figures use the site style in scripts/figstyle.py. Titles live in the article
captions, not inside the figures.

Usage
-----
    python make_figures.py                 # all figures it can legitimately make
    python make_figures.py --no-embed      # schematics only (no model load)

The first run loads BAAI/bge-m3 from the local Hugging Face cache and writes
embeddings_bge_m3.npz next to this file; later runs reuse that cache.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- #
# House style
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import (apply, tidy, title_case, INK, SLATE, RUST, DIM, GRID, MUTED,  # noqa: E402
                      LIGHT, SLATE_TINT, RUST_TINT, PAPER, CMAP, CMAP_DIVERGING, CATEGORICAL)

PALETTE = [DIM, SLATE, MUTED, LIGHT, MUTED]
FILL = {SLATE: SLATE_TINT, RUST: RUST_TINT}          # box fill by edge colour; PAPER otherwise
CODE_CHARS = set("_.=[]:<>")


def tc(text: str) -> str:
    """title_case() for figure text, leaving code-like words (a_b, x.y, --flag) as written."""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("["):          # tensor shapes such as [batch, seq, 1024]
            out.append(line); continue
        orig, new = line.split(" "), title_case(line).split(" ")
        out.append(" ".join(o if (CODE_CHARS & set(o) or o.startswith("--")) else n
                            for o, n in zip(orig, new)))
    return "\n".join(out)

SENTENCES = [
    "The golden rays of the setting sun painted the sky in shades of orange and pink.",
    "A sudden downpour drenched the city streets, leaving behind the fresh scent of rain.",
    "She hesitated before stepping onto the stage, her heartbeat quickening with anticipation.",
    "The discovery of a hidden passage behind the old bookshelf changed everything.",
    "Despite years of effort, scientists have yet to fully understand the nature of dark matter.",
    "In the stillness of the forest, the only sound was the distant hoot of an owl.",
    "A friendly stray cat began following him home every evening, as if adopting him.",
    "The ancient manuscript was covered in cryptic symbols that no one could decipher.",
    "After months of training, he finally completed the marathon, exhausted but triumphant.",
    "The artificial intelligence system learned to recognize human emotions with surprising accuracy.",
    "He glanced at his watch, realizing he had been lost in thought for nearly an hour.",
    "The cat curled up on the warm windowsill, watching the raindrops race down the glass.",
    "The detective examined the letter carefully, noting the peculiar phrasing of certain words.",
    "The city skyline shimmered in the golden light of the setting sun.",
    "A sudden gust of wind sent autumn leaves swirling in a mesmerizing dance.",
]

RECEIPT_SENTENCES = [
    "WALMART SUPERSTORE #1234",
    "500g Organic Canned Tomatoes",
    "2.99",
    "Thank you, your total is: $32.66",
    "VISA ****1234",
    "12/15/2024 3:45 PM",
]

# Facts taken from BAAI/bge-m3 config.json and from the MultiTaskModel source.
HIDDEN = 1024
N_LAYERS = 24
N_HEADS = 16
N_CLASSES = 4
N_NER = 5
BOTTLENECK = 128
# Real BGE-M3 tokenizer output for "500g Organic Canned Tomatoes" (10 subword ids
# including <s> and </s>); used only to make the shape annotations concrete.
EXAMPLE_TEXT = "500g Organic Canned Tomatoes"
EXAMPLE_LEN = 10


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    print("  wrote", out.name)


# --------------------------------------------------------------------------- #
# Diagram primitives (schematics)
# --------------------------------------------------------------------------- #
def new_canvas(w: float, h: float):
    """Axes spanning 0..1 in both directions with no decoration."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def _vfrac(ax, points: float) -> float:
    """Convert a vertical distance in typographic points to axes fraction.

    The canvases span 0..1 in y, so this is just points / (72 * figure height).
    Doing the conversion properly is what keeps stacked text from colliding on
    short, wide figures.
    """
    return points / (72.0 * ax.figure.get_size_inches()[1])


def _hfrac(ax, points: float) -> float:
    """Convert a horizontal distance in points to axes fraction."""
    return points / (72.0 * ax.figure.get_size_inches()[0])


def _fits(ax, text: str, size: float, w: float, name: str) -> None:
    """Warn if a string is wider than the box meant to hold it.

    Inter averages a little over half an em per character; 0.56 em is a safe
    estimate for the mixed-case strings used here.
    """
    need = _hfrac(ax, len(text) * size * 0.56) + _hfrac(ax, 8)
    if need > w:
        print(f"    ! '{text}' may overflow its box in {name} "
              f"({need:.3f} > {w:.3f})")


def box(ax, cx, cy, w, h, title, detail=None, color=SLATE, fill=None,
        title_size=10.0, detail_size=8.2, lw=1.3, title_weight="semibold",
        name=""):
    """Rounded box centred on (cx, cy) with a title and optional detail lines.

    Title and detail lines are stacked as one block and centred vertically, with
    line heights measured in points so the spacing is correct whatever the
    figure's aspect ratio.
    """
    title = tc(title)
    if detail:
        detail = [tc(d) for d in (detail if isinstance(detail, (list, tuple)) else [detail])]
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=lw, edgecolor=color, facecolor=FILL.get(color, PAPER), zorder=2,
    )
    ax.add_patch(patch)
    _fits(ax, title, title_size, w, name)

    lines = []
    if detail:
        lines = list(detail) if isinstance(detail, (list, tuple)) else [detail]
        for line in lines:
            _fits(ax, line, detail_size, w, name)

    title_h = _vfrac(ax, title_size * 1.30)
    detail_h = _vfrac(ax, detail_size * 1.45)
    gap_h = _vfrac(ax, 2.0) if lines else 0.0
    block = title_h + gap_h + len(lines) * detail_h
    if block > h:
        print(f"    ! text block taller than box for '{title}' in {name} "
              f"({block:.3f} > {h:.3f})")

    top = cy + block / 2
    ax.text(cx, top - title_h / 2, title, ha="center", va="center",
            fontsize=title_size, fontweight=title_weight, color=INK, zorder=3)
    for i, line in enumerate(lines):
        ax.text(cx, top - title_h - gap_h - (i + 0.5) * detail_h, line,
                ha="center", va="center", fontsize=detail_size, color=MUTED,
                zorder=3)
    return patch


def arrow(ax, start, end, color=SLATE, lw=1.5, style="-|>", rad=0.0, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13, linewidth=lw,
        color=color, alpha=alpha, zorder=1,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0,
    ))


def subtitle(ax, text, pad_pt: float = 5.0, size: float = 8.4) -> None:
    """Italic note in the gap between the axes and its title.

    Offsetting in points (rather than axes fraction) is what keeps the note
    below the title on short figures, where an axes fraction of 1.05 can land
    above it.
    """
    from matplotlib.transforms import ScaledTranslation
    tr = ax.transAxes + ScaledTranslation(0, pad_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0.5, 1.0, text, transform=tr, ha="center", va="bottom",
            fontsize=size, color=MUTED)


# Title pad for axes that also carry a subtitle: clears the note beneath it.
SUB_TITLE_PAD = 22


# --------------------------------------------------------------------------- #
# 1. System pipeline (schematic)
# --------------------------------------------------------------------------- #
def fig_system_pipeline(out: Path) -> None:
    """Top-to-bottom flow, sized for a narrow scrolling column."""
    fig, ax = new_canvas(6.2, 8.4)
    stages = [
        ("Receipt Image", ["photo or upload, JPEG / PNG"], PALETTE[2]),
        ("OCR", ["Tesseract --psm 6"], SLATE),
        ("Segmentation", ["split lines on newlines and periods"], SLATE),
        ("BGE-M3 Encoder", [f"{N_LAYERS} layers, frozen, {HIDDEN}-d states"], DIM),
        ("Task Heads", ["classification + NER"], RUST),
        ("Structured Data", ["line class, name and price spans"], PALETTE[2]),
    ]
    n = len(stages)
    cx, w, h = 0.40, 0.62, 0.105
    top, bottom = 0.93, 0.07
    step = (top - bottom) / (n - 1)
    ys = [top - i * step for i in range(n)]
    for i, (title, detail, color) in enumerate(stages):
        box(ax, cx, ys[i], w, h, title, detail, color=color, title_size=11.0,
            detail_size=9.0, name="system_pipeline")
        if i:
            arrow(ax, (cx, ys[i - 1] - h / 2 - 0.004), (cx, ys[i] + h / 2 + 0.004),
                  color=LIGHT, lw=1.6)

    # Stage groupings as brackets on the right, aligned to the boxes they cover.
    groups = [(0, 1, "OCR"), (2, 2, "Preprocessing"), (3, 4, "Multi-Task Model")]
    xb = cx + w / 2 + 0.05
    for a, b, label in groups:
        ya, yb = ys[a] + h / 2, ys[b] - h / 2
        ax.plot([xb, xb], [yb, ya], color=LIGHT, lw=3, solid_capstyle="butt", zorder=0)
        ax.text(xb + 0.03, (ya + yb) / 2, label, ha="left", va="center",
                fontsize=10, color=MUTED)
    save(fig, out)


# --------------------------------------------------------------------------- #
# 2. OCR pipeline (schematic)
# --------------------------------------------------------------------------- #
def fig_ocr_pipeline(out: Path) -> None:
    fig, ax = new_canvas(12.6, 4.4)
    steps = [
        ("Receipt Image", ["cv2.imread"], PALETTE[2]),
        ("Grayscale", ["cv2.cvtColor", "BGR2GRAY"], SLATE),
        ("Denoise", ["fastNlMeansDenoising", "h = 10"], SLATE),
        ("Threshold", ["Otsu binarisation", "THRESH_BINARY"], SLATE),
        ("Text Extraction", ["pytesseract", "image_to_string, psm 6"], PALETTE[0]),
        ("Line Segmentation", ["split on newline,", "then on period"], RUST),
        ("Sentence List", ["input to the", "transformer"], PALETTE[4]),
    ]
    # Two rows: 4 then 3, so no box is narrower than its longest detail line.
    rows = [steps[:4], steps[4:]]
    w, h, gap = 0.205, 0.28, 0.050
    ys = [0.70, 0.26]
    centres = []

    for r, row in enumerate(rows):
        span = len(row) * w + (len(row) - 1) * gap
        x0 = (1 - span) / 2 + w / 2
        row_centres = []
        for i, (title, detail, color) in enumerate(row):
            cx = x0 + i * (w + gap)
            row_centres.append(cx)
            box(ax, cx, ys[r], w, h, title, detail, color=color,
                title_size=9.8, detail_size=7.9, name="ocr_pipeline")
            if i:
                arrow(ax, (row_centres[i - 1] + w / 2 + 0.005, ys[r]),
                      (cx - w / 2 - 0.005, ys[r]), color=LIGHT, lw=1.5)
        centres.append(row_centres)

    # Wrap arrow from the end of row 1 down to the start of row 2.
    xe = centres[0][-1]
    xs = centres[1][0]
    ymid = (ys[0] + ys[1]) / 2
    ax.plot([xe, xe], [ys[0] - h / 2 - 0.005, ymid], color=LIGHT, lw=1.5,
            zorder=1)
    ax.plot([xe, xs], [ymid, ymid], color=LIGHT, lw=1.5, zorder=1)
    arrow(ax, (xs, ymid), (xs, ys[1] + h / 2 + 0.005), color=LIGHT, lw=1.5)

    ax.text(0.5, ymid + 0.028, "Preprocessed Image", ha="center", va="bottom",
            fontsize=8.2, color=MUTED)

    save(fig, out)


# --------------------------------------------------------------------------- #
# 3. Architecture diagram (schematic)
# --------------------------------------------------------------------------- #
def fig_architecture(out: Path) -> None:
    fig, ax = new_canvas(10.6, 7.6)

    NAME = "architecture_diagram"
    xl, xr, xc = 0.255, 0.745, 0.5
    # Row centres and heights, chosen so every inter-row gap clears the arrows.
    y_in, h_in = 0.893, 0.078
    y_bb, h_bb = 0.735, 0.140
    y_tok, h_tok = 0.552, 0.105
    y_head, h_head = 0.360, 0.160
    y_out, h_out = 0.148, 0.105
    wide, w = 0.46, 0.30

    box(ax, xc, y_in, 0.34, h_in, "Input Tokens", color=PALETTE[2],
        title_size=10.5, name=NAME)

    box(ax, xc, y_bb, wide, h_bb, "BGE-M3 Transformer Backbone",
        [f"XLM-RoBERTa, {N_LAYERS} layers, {N_HEADS} heads, "
         f"hidden size {HIDDEN}", "frozen during training"],
        color=PALETTE[0], title_size=11.0, detail_size=8.6, name=NAME)

    box(ax, xl, y_tok, w, h_tok, "CLS Token", [f"[batch, {HIDDEN}]"],
        color=PALETTE[2], title_size=10.0, name=NAME)
    box(ax, xr, y_tok, w, h_tok, "All Token States", [f"[batch, seq, {HIDDEN}]"],
        color=PALETTE[2], title_size=10.0, name=NAME)

    box(ax, xl, y_head, w, h_head, "Classification Head",
        [f"Linear {HIDDEN} to {BOTTLENECK}", "ReLU, dropout 0.1",
         f"Linear {BOTTLENECK} to {N_CLASSES}"],
        color=SLATE, title_size=10.0, detail_size=8.4, name=NAME)
    box(ax, xr, y_head, w, h_head, "NER Head",
        [f"Linear {HIDDEN} to {N_NER}", "applied at every token"],
        color=SLATE, title_size=10.0, detail_size=8.4, name=NAME)

    box(ax, xl, y_out, w, h_out, "Classification Logits",
        [f"[batch, {N_CLASSES}]"],
        color=RUST, title_size=10.0, detail_size=8.2, name=NAME)
    box(ax, xr, y_out, w, h_out, "NER Logits",
        [f"[batch, seq, {N_NER}]"],
        color=RUST, title_size=10.0, detail_size=8.2, name=NAME)

    pad = 0.005
    arrow(ax, (xc, y_in - h_in / 2 - pad), (xc, y_bb + h_bb / 2 + pad))
    for x in (xl, xr):
        arrow(ax, (xc, y_bb - h_bb / 2 - pad), (x, y_tok + h_tok / 2 + pad))
        arrow(ax, (x, y_tok - h_tok / 2 - pad), (x, y_head + h_head / 2 + pad))
        arrow(ax, (x, y_head - h_head / 2 - pad), (x, y_out + h_out / 2 + pad))

    y_edge = (y_tok - h_tok / 2 + y_head + h_head / 2) / 2
    ax.text(xl + 0.014, y_edge, "last_hidden_state[:, 0, :]", ha="left",
            va="center", fontsize=8.0, color=MUTED)
    ax.text(xr + 0.014, y_edge, "last_hidden_state", ha="left", va="center",
            fontsize=8.0, color=MUTED)

    # Both heads are the trainable part of the network; the backbone is not.
    # Pinned beside the head boxes so the word labels the layer, not the arrow.
    for x in (xl, xr):
        ax.text(x + w / 2 + 0.014, y_head, "Trainable", ha="left", va="center",
                fontsize=8.4, color=RUST)

    ax.text(xl, 0.048, "Sentence Class\n(ITEM, ITEM_OPTION, SUBTOTAL, TOTAL)",
            ha="center", va="top", fontsize=8.4, color=MUTED, linespacing=1.5)
    ax.text(xr, 0.048, "BIO Tags\n(O, B/I-NAME, B/I-PRICE)",
            ha="center", va="top", fontsize=8.4, color=MUTED, linespacing=1.5)

    save(fig, out)


# --------------------------------------------------------------------------- #
# 4. Forward pass flow (schematic)
# --------------------------------------------------------------------------- #
def fig_forward_pass(out: Path) -> None:
    fig, ax = new_canvas(11.4, 7.8)

    NAME = "forward_pass_flow"
    xc, xl, xr = 0.5, 0.255, 0.745
    w_wide, w = 0.50, 0.31

    y = [0.900, 0.770, 0.615, 0.445, 0.285, 0.130]
    hh = [0.082, 0.100, 0.130, 0.130, 0.130, 0.098]

    box(ax, xc, y[0], w_wide, hh[0], f'Input text:  "{EXAMPLE_TEXT}"',
        color=PALETTE[2], title_size=10.0, title_weight="normal", name=NAME)

    box(ax, xc, y[1], w_wide, hh[1], "Tokenizer",
        [f"input_ids [1, {EXAMPLE_LEN}]    attention_mask [1, {EXAMPLE_LEN}]"],
        color=PALETTE[2], title_size=10.0, detail_size=8.4, name=NAME)

    box(ax, xc, y[2], w_wide, hh[2], "self.base_model(input_ids, attention_mask)",
        [f"last_hidden_state  [1, {EXAMPLE_LEN}, {HIDDEN}]",
         "no gradient: backbone frozen"],
        color=PALETTE[0], title_size=10.2, detail_size=8.4, name=NAME)

    box(ax, xl, y[3], w, hh[3], "cls_embedding",
        ["last_hidden_state[:, 0, :]", f"[1, {HIDDEN}]"],
        color=SLATE, title_size=10.0, detail_size=8.4, name=NAME)
    box(ax, xr, y[3], w, hh[3], "full sequence",
        ["all token states", f"[1, {EXAMPLE_LEN}, {HIDDEN}]"],
        color=SLATE, title_size=10.0, detail_size=8.4, name=NAME)

    box(ax, xl, y[4], w, hh[4], "self.classifier",
        ["Linear -> ReLU -> Linear", f"logits [1, {N_CLASSES}]"],
        color=SLATE, title_size=10.0, detail_size=8.4, name=NAME)
    box(ax, xr, y[4], w, hh[4], "self.ner_head",
        [f"Linear {HIDDEN} -> {N_NER}",
         f"logits [1, {EXAMPLE_LEN}, {N_NER}]"],
        color=SLATE, title_size=10.0, detail_size=8.4, name=NAME)

    box(ax, xl, y[5], w, hh[5], "CrossEntropyLoss", ["class_loss"],
        color=RUST, title_size=9.8, detail_size=8.2, name=NAME)
    box(ax, xr, y[5], w, hh[5], "CrossEntropyLoss", ["ner_loss (flattened)"],
        color=RUST, title_size=9.8, detail_size=8.2, name=NAME)

    pad = 0.005
    arrow(ax, (xc, y[0] - hh[0] / 2 - pad), (xc, y[1] + hh[1] / 2 + pad))
    arrow(ax, (xc, y[1] - hh[1] / 2 - pad), (xc, y[2] + hh[2] / 2 + pad))
    for x in (xl, xr):
        arrow(ax, (xc, y[2] - hh[2] / 2 - pad), (x, y[3] + hh[3] / 2 + pad))
        arrow(ax, (x, y[3] - hh[3] / 2 - pad), (x, y[4] + hh[4] / 2 + pad))
        arrow(ax, (x, y[4] - hh[4] / 2 - pad), (x, y[5] + hh[5] / 2 + pad))

    # Joint loss: brackets joining the two loss boxes to a summed objective.
    y_join = 0.048
    ax.plot([xl, xl], [y[5] - hh[5] / 2 - pad, y_join], color=RUST, lw=1.4,
            zorder=1)
    ax.plot([xr, xr], [y[5] - hh[5] / 2 - pad, y_join], color=RUST, lw=1.4,
            zorder=1)
    ax.plot([xl, xr], [y_join, y_join], color=RUST, lw=1.4, zorder=1)
    ax.text(xc, y_join - 0.030,
            "total_loss = class_loss + ner_loss   ->   backward(),  "
            "clip_grad_norm_(max_norm = 1.0),  step()",
            ha="center", va="center", fontsize=8.8, color=RUST)

    save(fig, out)


# --------------------------------------------------------------------------- #
# Real-data figures
# --------------------------------------------------------------------------- #
CACHE = HERE / "embeddings_bge_m3.npz"


def load_embeddings(allow_encode: bool = True):
    """Return (embeddings, receipt_embeddings). Encodes with BGE-M3 if needed."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["embeddings"], d["receipt_embeddings"]
    if not allow_encode:
        return None, None

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import random
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    print("  loading BAAI/bge-m3 ...")
    model = SentenceTransformer("BAAI/bge-m3")
    emb = model.encode(SENTENCES, normalize_embeddings=True,
                       convert_to_tensor=True).cpu().numpy()
    rec = model.encode(RECEIPT_SENTENCES, normalize_embeddings=True,
                       convert_to_tensor=True).cpu().numpy()
    np.savez_compressed(CACHE, embeddings=emb, receipt_embeddings=rec,
                        sentences=np.array(SENTENCES, dtype=object),
                        receipt_sentences=np.array(RECEIPT_SENTENCES, dtype=object))
    print("  cached", CACHE.name)
    return emb, rec


def fig_embedding_heatmap(emb, out: Path) -> None:
    """REAL DATA: the first sentence's 1024-d BGE-M3 embedding."""
    v = emb[0]
    # A handful of dimensions are several times larger than the rest, so the
    # full range washes the strip out. Clip the colour scale at the 99th
    # percentile of |value| and say so, rather than hiding the choice.
    lim = float(np.percentile(np.abs(v), 99))

    fig, ax = plt.subplots(figsize=(13.0, 1.85))
    im = ax.imshow(v.reshape(1, -1), cmap=CMAP_DIVERGING, aspect="auto",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Embedding Dimension")
    ax.set_xlim(-0.5, len(v) - 0.5)
    ax.set_xticks([0, 255, 511, 767, 1023])
    ax.set_xticklabels(["1", "256", "512", "768", "1024"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(colors=INK, labelsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.010)
    cbar.set_label("Value", fontsize=9, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=INK)
    cbar.outline.set_visible(False)

    save(fig, out)


def fig_similarity_matrix(emb, out: Path) -> None:
    """REAL DATA: cosine similarity between the 15 BGE-M3 embeddings."""
    sim = emb @ emb.T
    n = len(sim)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(sim, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(n), [str(i + 1) for i in range(n)])
    ax.set_yticks(range(n), [str(i + 1) for i in range(n)])
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Sentence Index")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=INK, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.1, color="white" if sim[i, j] > 0.62 else INK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.022)
    cbar.set_label("Cosine Similarity", fontsize=9, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=INK)
    cbar.outline.set_visible(False)

    save(fig, out)


def fig_tsne(emb, out: Path) -> None:
    """REAL DATA: k-means over the BGE-M3 embeddings, laid out with t-SNE."""
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    n_clusters = 4
    labels = KMeans(n_clusters=n_clusters, random_state=42,
                    n_init=10).fit_predict(emb)
    xy = TSNE(n_components=2, perplexity=min(5, len(emb) - 1),
              random_state=42, init="pca").fit_transform(emb)

    # The project palette's first three entries are all dark blue-greys, which
    # is fine for sequential work but unreadable as categories; pick four
    # house colours that separate cleanly instead.
    cluster_colors = CATEGORICAL[:4]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for k in range(n_clusters):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=150,
                   color=cluster_colors[k % len(cluster_colors)],
                   edgecolors="white", linewidth=1.4, alpha=0.95, zorder=3,
                   label=f"Cluster {k}")
    # Offset the index labels in points, not data units: the two axes span very
    # different ranges, so a single data-unit offset drifts labels off their
    # markers and toward their neighbours.
    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                    xytext=(9, 5), fontsize=8.6, color=INK, alpha=0.9,
                    ha="left", va="bottom", zorder=4)

    ax.margins(0.12)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    tidy(ax)
    leg = ax.legend(frameon=False, loc="lower left", fontsize=9, title="k-Means")
    leg.get_title().set_fontsize(9)
    save(fig, out)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "assets"))
    ap.add_argument("--no-embed", action="store_true",
                    help="skip figures that need BGE-M3 inference")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply()

    print("Schematics:")
    fig_system_pipeline(out / "system_pipeline.png")
    fig_ocr_pipeline(out / "ocr_pipeline.png")
    fig_architecture(out / "architecture_diagram.png")
    fig_forward_pass(out / "forward_pass_flow.png")

    print("Real BGE-M3 output:")
    emb, _ = load_embeddings(allow_encode=not args.no_embed)
    if emb is None:
        print("  skipped (no cached embeddings and --no-embed given)")
    else:
        fig_embedding_heatmap(emb, out / "embedding_heatmap.png")
        fig_similarity_matrix(emb, out / "similarity_matrix.png")
        fig_tsne(emb, out / "tsne_clustering.png")


if __name__ == "__main__":
    main()
