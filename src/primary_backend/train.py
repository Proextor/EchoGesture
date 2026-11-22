"""Train EchoNet on landmark sequences.

    python train.py --train ../data/train.npz --val ../data/val.npz --epochs 300
Saves models/echonet.pt (best validation top-1) and models/classes.json

Final ensemble (train+val, fixed schedule, one model per seed):
    for s in 1 2 3 4 5; do python train.py --train ../data/train.npz --also-train ../data/val.npz \
        --val ../data/val.npz --save-last --seed $s --out echonet_s$s.pt; done
(with --also-train the printed val accuracy is on data the model has seen - ignore it.)
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from .augment import random_augment, random_resample
from .config import MODEL_DIR, CLASSES_PATH, SEED, load_vocab
from .features import load_sequences, resample, to_model_input
from .model import EchoNet


def load(path, idx):
    seqs, y = load_sequences(path)
    keep = [i for i, c in enumerate(y) if c in idx]
    return [seqs[i] for i in keep], np.array([idx[y[i]] for i in keep])


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, seqs, y):
        self.seqs, self.y = seqs, y

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = random_augment(random_resample(self.seqs[i]))
        return torch.from_numpy(to_model_input(s)), int(self.y[i])


def fixed(seqs):
    return torch.from_numpy(np.stack([to_model_input(resample(s)) for s in seqs]))


def top1(net, X, y):
    net.eval()
    with torch.no_grad():
        return (net(X).argmax(1).numpy() == y).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--also-train", nargs="*", default=[], help="extra npz files merged into the training set")
    ap.add_argument("--out", default="echonet.pt", help="checkpoint file name inside models/")
    ap.add_argument("--save-last", action="store_true", help="keep the final epoch instead of the best-val epoch")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    vocab = load_vocab()
    classes = vocab or sorted(set(load_sequences(a.train)[1].tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    tr_s, tr_y = load(a.train, idx)
    for extra in a.also_train:
        es, ey = load(extra, idx)
        tr_s, tr_y = tr_s + es, np.concatenate([tr_y, ey])
    va_s, va_y = load(a.val, idx)
    Xv = fixed(va_s)
    print(f"{len(classes)} classes, {len(tr_s)} train / {len(va_s)} val clips")

    dl = torch.utils.data.DataLoader(TrainDS(tr_s, tr_y), batch_size=a.bs, shuffle=True, drop_last=True)
    net = EchoNet(len(classes))
    print("params:", sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = -1.0
    MODEL_DIR.mkdir(exist_ok=True)
    CLASSES_PATH.write_text(json.dumps(classes))
    for ep in range(a.epochs):
        net.train()
        for x, y in dl:
            opt.zero_grad()
            if a.mixup > 0:
                lam = float(np.random.beta(a.mixup, a.mixup))
                perm = torch.randperm(len(x))
                out = net(lam * x + (1 - lam) * x[perm])
                loss = lam * loss_fn(out, y) + (1 - lam) * loss_fn(out, y[perm])
            else:
                loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            sched.step()
        acc = top1(net, Xv, va_y)
        best = max(best, acc)
        if a.save_last or acc >= best:
            torch.save(net.state_dict(), MODEL_DIR / a.out)
        if (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep + 1:3d}  loss {loss.item():.3f}  val top1 {acc:.3f}  best {best:.3f}", flush=True)
    print("best val top1", round(float(best), 4))


if __name__ == "__main__":
    main()
