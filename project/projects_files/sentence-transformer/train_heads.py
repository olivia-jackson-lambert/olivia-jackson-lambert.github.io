"""
Train the multi-task heads on CORD-v2 receipt lines.

Implements the freezing strategy the write-up describes: the BGE-M3 backbone is
frozen, so its token states are computed once and cached, and only the two task
heads are trained. This is what makes the run tractable on a laptop.

  classification head   CLS state -> 1024 -> 128 -> 4 classes
  NER head              per-token state -> 5 BIO labels

Usage:  python train_heads.py
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

DATA = Path("/Users/oliviajackson/Documents/portfolio/projects/receipt-data/cord_multitask")
ART = Path("/Users/oliviajackson/Documents/portfolio/projects/receipt-data/artifacts")
MODEL = "BAAI/bge-m3"
MAXLEN, SEED, EPOCHS, BATCH = 32, 42, 30, 64
CLASS_NAMES = ["ITEM", "ITEM_OPTION", "SUBTOTAL", "TOTAL"]
NER_NAMES = ["O", "B-NAME", "I-NAME", "B-PRICE", "I-PRICE"]
PAD = -100


def encode(split: str, tok, backbone, dev):
    rows = json.loads((DATA / f"{split}.json").read_text())
    feats, ner, cls = [], [], []
    for i in range(0, len(rows), 32):
        chunk = rows[i:i + 32]
        enc = tok([r["words"] for r in chunk], is_split_into_words=True,
                  truncation=True, max_length=MAXLEN, padding="max_length",
                  return_tensors="pt")
        with torch.no_grad():
            h = backbone(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state
        feats.append(h.cpu().to(torch.float16))
        for j, r in enumerate(chunk):
            wid = enc.word_ids(j)
            tags, prev = [], None
            for w in wid:
                if w is None:
                    tags.append(PAD)                      # special / padding
                elif w != prev:
                    tags.append(r["ner"][w])              # first subword carries the tag
                else:
                    tags.append(PAD)                      # ignore continuation subwords
                prev = w
            ner.append(tags)
            cls.append(r["cls"])
    return (torch.cat(feats), torch.tensor(ner), torch.tensor(cls))


class Heads(nn.Module):
    def __init__(self, dim=1024, n_cls=4, n_ner=5):
        super().__init__()
        self.cls_head = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, n_cls))
        self.ner_head = nn.Linear(dim, n_ner)

    def forward(self, h):
        return self.cls_head(h[:, 0]), self.ner_head(h)


def evaluate(heads, H, Y_ner, Y_cls, dev, cfn, nfn):
    heads.eval()
    with torch.no_grad():
        cl, nl, cp, np_, nt = 0.0, 0.0, [], [], []
        for i in range(0, len(H), 256):
            h = H[i:i + 256].to(dev).float()
            yc, yn = Y_cls[i:i + 256].to(dev), Y_ner[i:i + 256].to(dev)
            lc, ln = heads(h)
            cl += cfn(lc, yc).item() * len(h)
            nl += nfn(ln.reshape(-1, ln.size(-1)), yn.reshape(-1)).item() * len(h)
            cp.append(lc.argmax(-1).cpu())
            m = yn != PAD
            np_.append(ln.argmax(-1)[m].cpu()); nt.append(yn[m].cpu())
    cp = torch.cat(cp); np_ = torch.cat(np_); nt = torch.cat(nt)
    return {
        "cls_loss": cl / len(H), "ner_loss": nl / len(H),
        "cls_acc": (cp == Y_cls).float().mean().item(),
        "cls_f1": f1_score(Y_cls.numpy(), cp.numpy(), average="weighted", zero_division=0),
        "ner_acc": (np_ == nt).float().mean().item(),
        "ner_f1": f1_score(nt.numpy(), np_.numpy(), average="macro", zero_division=0),
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(MODEL)
    backbone = AutoModel.from_pretrained(MODEL).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad = False                      # the freezing the write-up describes
    n_frozen = sum(p.numel() for p in backbone.parameters())
    print(f"backbone frozen: {n_frozen/1e6:.0f}M parameters, requires_grad=False")

    cache = ART / "encoded.pt"
    if cache.exists():
        d = torch.load(cache); splits = d["splits"]
        print("loaded cached encodings")
    else:
        splits = {}
        for s in ("train", "validation", "test"):
            t0 = time.time()
            splits[s] = encode(s, tok, backbone, dev)
            print(f"  encoded {s}: {tuple(splits[s][0].shape)} in {time.time()-t0:.0f}s")
        torch.save({"splits": splits}, cache)

    heads = Heads().to(dev)
    n_train = sum(p.numel() for p in heads.parameters())
    print(f"trainable head parameters: {n_train/1e3:.0f}K "
          f"({100*n_train/(n_train+n_frozen):.3f}% of the whole model)\n")

    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3)
    cfn = nn.CrossEntropyLoss()
    nfn = nn.CrossEntropyLoss(ignore_index=PAD)

    Htr, Ntr, Ctr = splits["train"]
    hist = []
    for ep in range(1, EPOCHS + 1):
        heads.train()
        perm = torch.randperm(len(Htr))
        for i in range(0, len(Htr), BATCH):
            idx = perm[i:i + BATCH]
            h = Htr[idx].to(dev).float()
            yc, yn = Ctr[idx].to(dev), Ntr[idx].to(dev)
            lc, ln = heads(h)
            loss = cfn(lc, yc) + nfn(ln.reshape(-1, ln.size(-1)), yn.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        tr = evaluate(heads, Htr, Ntr, Ctr, dev, cfn, nfn)
        va = evaluate(heads, *splits["validation"][::1][0:1] and
                      (splits["validation"][0], splits["validation"][1], splits["validation"][2]),
                      dev, cfn, nfn)
        hist.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                     **{f"val_{k}": v for k, v in va.items()}})
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:>2}  cls {tr['cls_loss']:.3f}/{va['cls_loss']:.3f}  "
                  f"acc {va['cls_acc']:.3f}  ner {tr['ner_loss']:.3f}/{va['ner_loss']:.3f}  "
                  f"ner_acc {va['ner_acc']:.3f}")

    te = evaluate(heads, *splits["test"], dev, cfn, nfn)
    print(f"\nTEST  cls_acc {te['cls_acc']:.4f}  cls_f1 {te['cls_f1']:.4f}  "
          f"ner_acc {te['ner_acc']:.4f}  ner_macro_f1 {te['ner_f1']:.4f}")

    json.dump(hist, open(ART / "history.json", "w"), indent=1)
    json.dump(te, open(ART / "test_metrics.json", "w"), indent=1)
    torch.save(heads.state_dict(), ART / "heads.pt")
    print("wrote", ART)


if __name__ == "__main__":
    main()
