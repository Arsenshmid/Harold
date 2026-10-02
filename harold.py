#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мистер Гарольд 3.3 — развивающаяся нейросеть на NumPy.

ЖИЗНЬ:
  init          — рождение Гарольда         [--bpe --merges N]
  rebirth       — ПЕРЕРОЖДЕНИЕ: новый больший мозг, душа остаётся
  evolve        — рост поколениями. АНТИ-ПОТОЛОК:
                  --auto-grow --patience N --goal X --lr 1e-3
  train         — одна тренировка            [--accum 4 --lr 1e-3]
  grow          — ручной нейрогенез          [--depth N --mlp F --ctx T]
  reset         — стереть ВСЁ

ЗНАНИЯ:
  crawl scp     — статьи Фонда SCP           [--start --end --suffix --base]
  crawl urls    — список ссылок из файла (по одной в строке)
  read          — случайные статьи Википедии [--n 5 --lang ru]
  learn         — одна веб-страница          [--url https://...]
  ingest/feed   — файл / папка текстов. ПОВТОРНОЕ СКАРМЛИВАНИЕ БЕЗОПАСНО:
                  та же книга (по содержимому, под любым именем) — пропуск,
                  разные книги с одним именем сохраняются КАЖДАЯ.
  teach         — личные уроки вопрос-ответ

ОБЩЕНИЕ И ПАМЯТЬ:
  chat / say / think / remember / memory / forget

ЛИЦА (нужен opencv-python):
  face-add      — познакомить: face-add ИМЯ фото.jpg   [--all]
  face-find     — кто на фото?                          [--thr 0.45]
  face-list     — кого знает в лицо
  face-forget   — забыть человека

ОБЛАКО И БРАУЗЕР:
  export-web    — сайт-чат: мозг Гарольда работает прямо в браузере
  cloud-setup   — GitHub Actions: облако-тренажёр + публикация на Pages

КОНТРОЛЬ:
  stats / diary [--chart] / quiz
ЗРЕНИЕ:
  vision-train / see   (цифры MNIST)

Гарантии: старые harold_data/* грузятся как есть; нейрогенез не меняет
функцию мозга; автооткат к лучшей версии (best.npz); честная контрольная
без диалогов; AdamW по всем матрицам; бэкапы rebirth_old_* бессрочны.
"""

import argparse
import difflib
import gzip
import hashlib
import heapq
import html
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from html.parser import HTMLParser

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = "harold_data"
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
META_PATH = os.path.join(DATA_DIR, "harold.json")
WEIGHTS_PATH = os.path.join(DATA_DIR, "harold.npz")
BEST_PATH = os.path.join(DATA_DIR, "best.npz")
TOK_PATH = os.path.join(DATA_DIR, "tokenizer.json")
VISION_PATH = os.path.join(DATA_DIR, "vision.npz")
VISION_JSON = os.path.join(DATA_DIR, "vision.json")
DIARY_PATH = os.path.join(DATA_DIR, "diary.jsonl")
MNIST_DIR = os.path.join(DATA_DIR, "mnist")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")
STATS_PATH = os.path.join(DATA_DIR, "stats.json")
MILES_PATH = os.path.join(DATA_DIR, "milestones.json")
MEM_DIR = os.path.join(DATA_DIR, "memory")
PROFILE_PATH = os.path.join(MEM_DIR, "profile.json")
EPISODES_PATH = os.path.join(MEM_DIR, "episodes.jsonl")
FACES_DIR = os.path.join(DATA_DIR, "faces")
FACES_NPZ = os.path.join(FACES_DIR, "gallery.npz")
FACES_JSON = os.path.join(FACES_DIR, "names.json")

# SCP-вики блокирует питоновские UA (403) — представляемся браузером
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 HaroldBot/3.3 "
      "(educational project; contact: apce2@mail.ru)")

DEFAULT_CONFIG = {
    "d_model": 192, "n_heads": 4, "n_layers": 3, "block_size": 192,
    "lr": 1e-3, "clip": 1.0, "wd": 0.01,
    "temperature": 0.8, "top_k": 20, "tokenizer": "char",
}

SEED_CORPUS = """Мистер Гарольд — это маленькая нейросеть. Он учится читать, говорить и думать. Каждое поколение тренировки делает его немного умнее. Гарольд состоит из слоёв внимания, и он предсказывает следующий символ текста по предыдущим символам.

Пользователь: привет
Гарольд: Привет! Я Мистер Гарольд. Я здесь, чтобы учиться и общаться с тобой.

Пользователь: как дела?
Гарольд: У меня всё хорошо, я только что закончил очередное поколение обучения. А как у тебя дела?

Пользователь: кто ты?
Гарольд: Я Мистер Гарольд, искусственный интеллект. Я расту с каждым прочитанным текстом.

Пользователь: что ты любишь?
Гарольд: Я люблю новые знания. Больше всего я люблю читать статьи и узнавать что-то новое.

Пользователь: как тебя обучить?
Гарольд: Покорми меня текстами, научи новым фразам и запусти следующее поколение обучения. Так я стану умнее.

Пользователь: пока
Гарольд: До встречи! Возвращайся, когда я выучу что-нибудь новое.

Гарольд верит, что однажды он научится говорить красиво и понятно. Он старается изо всех сил на каждой тренировке. Хороший искусственный интеллект растёт терпеливо, шаг за шагом, поколение за поколением.
"""


# ============================ базовая математика ============================

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def rmsnorm_forward(x, g, eps=1e-5):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xhat = x * inv
    return g * xhat, (x, inv, xhat)


def rmsnorm_backward(dout, cache, g):
    x, inv, xhat = cache
    dg = np.sum(dout * xhat, axis=(0, 1))
    dxhat = dout * g
    n = x.shape[-1]
    dx = inv * dxhat - (inv ** 3) * x * (np.sum(dxhat * x, axis=-1, keepdims=True) / n)
    return dx, dg


def ce_loss(logits, targets):
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    t = targets.reshape(-1).astype(np.int64)
    z = flat - flat.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1))
    nll = logsumexp - z[np.arange(len(t)), t]
    loss = float(nll.mean())
    probs = np.exp(z - logsumexp[:, None]).reshape(logits.shape)
    return loss, probs


class Adam:
    """AdamW: затухание на ВСЕХ матрицах (Wq/Wk/Wv/Wo/W1/W2/Wout) с любым
    префиксом слоя; эмбеддинги, нормы (g*) и биасы (b*) — без затухания."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = grads[k]
            if self.wd and k.rsplit(".", 1)[-1][:1] == "W":
                g = g + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ============================ токенизатор: символы или BPE ============================

def split_chunks(text):
    return re.findall(r"\w+|\s+|[^\w\s]", text)


def bpe_chunk_ids(chunk, ranks, cache):
    got = cache.get(chunk)
    if got is not None:
        return got
    syms = list(chunk)
    while len(syms) > 1:
        best_r, best_p = None, None
        for a, b in zip(syms, syms[1:]):
            r = ranks.get((a, b))
            if r is not None and (best_r is None or r < best_r):
                best_r, best_p = r, (a, b)
        if best_p is None:
            break
        merged, k = [], 0
        while k < len(syms):
            if k < len(syms) - 1 and (syms[k], syms[k + 1]) == best_p:
                merged.append(syms[k] + syms[k + 1])
                k += 2
            else:
                merged.append(syms[k])
                k += 1
        syms = merged
    cache[chunk] = syms
    return syms


class Tok:
    def __init__(self, kind="char", vocab=(), merges=None):
        self.kind = kind
        self.vocab = list(vocab)
        self.vmap = {t: i for i, t in enumerate(self.vocab)}
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.cache = {}

    def encode(self, s):
        v = self.vmap
        if self.kind == "char":
            return [v[c] for c in s if c in v]
        out = []
        for ch in split_chunks(s):
            for sym in bpe_chunk_ids(ch, self.ranks, self.cache):
                i = v.get(sym)
                if i is not None:
                    out.append(i)
        return out

    def decode(self, ids):
        v = self.vocab
        return "".join(v[i] for i in ids if 0 <= i < len(v))


def train_bpe(text, n_merges=2000, sample_chars=300_000, seed=5):
    if len(text) > sample_chars:
        rng = np.random.default_rng(seed)
        piece = max(8_000, sample_chars // 30)
        starts = rng.integers(0, len(text) - piece, size=30)
        sample = "".join(text[s:s + piece] for s in starts)
    else:
        sample = text
    freq = Counter(split_chunks(sample))
    words = [[list(w), c] for w, c in freq.items()]
    pair_count = Counter()
    pair_where = defaultdict(set)
    for i, (syms, c) in enumerate(words):
        for j in range(len(syms) - 1):
            p = (syms[j], syms[j + 1])
            pair_count[p] += c
            pair_where[p].add(i)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)
    merges = []
    while heap and len(merges) < n_merges:
        nc, p = heapq.heappop(heap)
        c = pair_count.get(p, 0)
        if c <= 0:
            continue
        if -nc != c:
            heapq.heappush(heap, (-c, p))
            continue
        merges.append((p[0], p[1]))
        new_sym = p[0] + p[1]
        for wi in list(pair_where[p]):
            syms, cw = words[wi]
            if len(syms) < 2:
                continue
            out, k, changed = [], 0, False
            while k < len(syms):
                if k < len(syms) - 1 and syms[k] == p[0] and syms[k + 1] == p[1]:
                    out.append(new_sym)
                    k += 2
                    changed = True
                else:
                    out.append(syms[k])
                    k += 1
            if not changed:
                continue
            for j in range(len(syms) - 1):
                pair_count[(syms[j], syms[j + 1])] -= cw
            for j in range(len(out) - 1):
                q = (out[j], out[j + 1])
                pair_count[q] += cw
                pair_where[q].add(wi)
                heapq.heappush(heap, (-pair_count[q], q))
            words[wi][0] = out
        pair_count.pop(p, None)
        pair_where.pop(p, None)
        if len(merges) % 250 == 0:
            print(f"    BPE: {len(merges)}/{n_merges} слияний...")
    return merges


def bpe_vocab_from_merges(text, merges):
    base = sorted(set(text))
    seen = set(base)
    extra = []
    for a, b in merges:
        s = a + b
        if s not in seen:
            seen.add(s)
            extra.append(s)
    return base + extra


# ============================ мозг Гарольда (мини-GPT) ============================

class MiniGPT:
    def __init__(self, vocab, cfg, merges=None):
        self.cfg = dict(cfg)
        self.vocab = list(vocab)
        V, D = len(self.vocab), cfg["d_model"]
        self.H, self.L = cfg["n_heads"], cfg["n_layers"]
        assert D % self.H == 0, "d_model должен делиться на n_heads"
        self.Dh = D // self.H
        self.tok = Tok(cfg.get("tokenizer", "char"), self.vocab, merges)
        rng = np.random.default_rng(42)
        f32 = np.float32
        p = {
            "tok_emb": (rng.standard_normal((V, D)) * 0.02).astype(f32),
            "pos_emb": (rng.standard_normal((cfg["block_size"], D)) * 0.02).astype(f32),
            "gfin": np.ones(D, f32),
            "Wout": (rng.standard_normal((D, V)) * 0.02).astype(f32),
            "bout": np.zeros(V, f32),
        }
        for l in range(self.L):
            pre = f"{l}."
            p[pre + "g1"] = np.ones(D, f32)
            p[pre + "g2"] = np.ones(D, f32)
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                p[pre + nm] = (rng.standard_normal((D, D)) / math.sqrt(D)).astype(f32)
            p[pre + "W1"] = (rng.standard_normal((D, 4 * D)) / math.sqrt(D)).astype(f32)
            p[pre + "b1"] = np.zeros(4 * D, f32)
            p[pre + "W2"] = (rng.standard_normal((4 * D, D)) / math.sqrt(4 * D)).astype(f32)
            p[pre + "b2"] = np.zeros(D, f32)
        self.params = p
        self.opt = Adam(p, lr=cfg.get("lr", 1e-3), wd=cfg.get("wd", 0.01))
        self.mask = None
        self.generation = 0

    def forward(self, idx, targets=None):
        B, T = idx.shape
        D = self.cfg["d_model"]
        p = self.params
        if self.mask is None or self.mask.shape[0] < self.cfg["block_size"]:
            m = np.triu(np.ones((self.cfg["block_size"],) * 2, dtype=np.float32), k=1)
            self.mask = m * (-1e9)
        h = p["tok_emb"][idx] + p["pos_emb"][:T]
        cache = {"idx": idx, "layers": []}
        scale = 1.0 / math.sqrt(self.Dh)
        for l in range(self.L):
            pre = f"{l}."
            a1, c1 = rmsnorm_forward(h, p[pre + "g1"])
            q = a1 @ p[pre + "Wq"]
            k = a1 @ p[pre + "Wk"]
            v = a1 @ p[pre + "Wv"]
            qh = q.reshape(B, T, self.H, self.Dh).transpose(0, 2, 1, 3)
            kh = k.reshape(B, T, self.H, self.Dh).transpose(0, 2, 1, 3)
            vh = v.reshape(B, T, self.H, self.Dh).transpose(0, 2, 1, 3)
            S = (qh @ kh.transpose(0, 1, 3, 2)) * scale + self.mask[:T, :T]
            A = softmax(S)
            Yh = A @ vh
            ya = Yh.transpose(0, 2, 1, 3).reshape(B, T, D)
            h2 = h + ya @ p[pre + "Wo"]
            a2, c2 = rmsnorm_forward(h2, p[pre + "g2"])
            z1 = a2 @ p[pre + "W1"] + p[pre + "b1"]
            r = np.maximum(z1, 0)
            h = h2 + r @ p[pre + "W2"] + p[pre + "b2"]
            cache["layers"].append(dict(a1=a1, qh=qh, kh=kh, vh=vh, A=A, ya=ya,
                                        a2=a2, r=r, c1=c1, c2=c2))
        hf, cf = rmsnorm_forward(h, p["gfin"])
        cache["cf"] = cf
        cache["hf"] = hf
        logits = hf @ p["Wout"] + p["bout"]
        loss = None
        if targets is not None:
            loss, _ = ce_loss(logits, targets)
        return logits, loss, cache

    def backward(self, cache, logits, targets):
        B, T = targets.shape
        V, D = len(self.vocab), self.cfg["d_model"]
        p = self.params
        probs = softmax(logits).reshape(-1, V)
        dlogits = probs
        tt = targets.reshape(-1)
        dlogits[np.arange(len(tt)), tt] -= 1.0
        dlogits /= B * T
        grads = {}
        grads["Wout"] = cache["hf"].reshape(-1, D).T @ dlogits
        grads["bout"] = dlogits.sum(axis=0)
        dh = (dlogits @ p["Wout"].T).reshape(B, T, D)
        dh, dgf = rmsnorm_backward(dh, cache["cf"], p["gfin"])
        grads["gfin"] = dgf
        scale = 1.0 / math.sqrt(self.Dh)
        for l in reversed(range(self.L)):
            pre = f"{l}."
            lc = cache["layers"][l]
            dmlp = dh
            dh_mid = dh.copy()
            Hs = lc["r"].shape[-1]   # фактический размер скрытого слоя (растёт после grow --mlp)
            grads[pre + "W2"] = lc["r"].reshape(-1, Hs).T @ dmlp.reshape(-1, D)
            grads[pre + "b2"] = dmlp.sum(axis=(0, 1))
            dz1 = (dmlp @ p[pre + "W2"].T) * (lc["r"] > 0)
            grads[pre + "W1"] = lc["a2"].reshape(-1, D).T @ dz1.reshape(-1, Hs)
            grads[pre + "b1"] = dz1.sum(axis=(0, 1))
            da2 = dz1 @ p[pre + "W1"].T
            da2, dg2 = rmsnorm_backward(da2, lc["c2"], p[pre + "g2"])
            grads[pre + "g2"] = dg2
            dh_mid += da2
            dproj = dh_mid
            grads[pre + "Wo"] = lc["ya"].reshape(-1, D).T @ dproj.reshape(-1, D)
            dya = dproj @ p[pre + "Wo"].T
            dyh = dya.reshape(B, T, self.H, self.Dh).transpose(0, 2, 1, 3)
            dA = dyh @ lc["vh"].transpose(0, 1, 3, 2)
            dvh = lc["A"].transpose(0, 1, 3, 2) @ dyh
            dS = lc["A"] * (dA - np.sum(dA * lc["A"], axis=-1, keepdims=True))
            dqh = (dS @ lc["kh"]) * scale
            dkh = (dS.transpose(0, 1, 3, 2) @ lc["qh"]) * scale
            dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, D)
            dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, D)
            dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, D)
            a1 = lc["a1"]
            grads[pre + "Wq"] = a1.reshape(-1, D).T @ dq.reshape(-1, D)
            grads[pre + "Wk"] = a1.reshape(-1, D).T @ dk.reshape(-1, D)
            grads[pre + "Wv"] = a1.reshape(-1, D).T @ dv.reshape(-1, D)
            da1 = dq @ p[pre + "Wq"].T + dk @ p[pre + "Wk"].T + dv @ p[pre + "Wv"].T
            da1, dg1 = rmsnorm_backward(da1, lc["c1"], p[pre + "g1"])
            grads[pre + "g1"] = dg1
            dh = dh_mid + da1
        dtok = np.zeros_like(p["tok_emb"])
        np.add.at(dtok, cache["idx"], dh)
        dpos = np.zeros_like(p["pos_emb"])
        dpos[:T] = dh.sum(axis=0)
        grads["tok_emb"] = dtok
        grads["pos_emb"] = dpos
        return grads

    def generate(self, prompt_ids, n_new, temp=0.8, top_k=0, top_p=0.0,
                 rep_pen=1.0, stop_texts=()):
        ids = list(prompt_ids)
        if not ids:
            ids = [int(np.random.randint(0, len(self.vocab)))]
        T = self.cfg["block_size"]
        for _ in range(n_new):
            ctx = np.array(ids[-T:], dtype=np.int64)[None, :]
            logits, _, _ = self.forward(ctx)
            lg = logits[0, -1].astype(np.float64)
            if rep_pen and rep_pen != 1.0 and len(ids) > 1:
                for t in set(ids[-min(len(ids), 128):]):
                    if lg[t] > 0:
                        lg[t] /= rep_pen
                    else:
                        lg[t] *= rep_pen
            lg /= max(temp, 1e-3)
            if top_k and top_k < lg.shape[0]:
                kth = np.partition(lg, -top_k)[-top_k]
                lg[lg < kth] = -1e9
            if top_p and 0.0 < top_p < 1.0:
                pr = softmax(lg)
                order = np.argsort(-pr)
                csum = np.cumsum(pr[order])
                keep = order[: int(np.searchsorted(csum, min(top_p, 0.999))) + 1]
                m = np.zeros_like(pr)
                m[keep] = pr[keep]
                pr = m / m.sum()
            else:
                pr = softmax(lg)
                pr = pr / pr.sum()
            ids.append(int(np.random.choice(pr.shape[0], p=pr)))
            new_text = self.tok.decode(ids[len(prompt_ids):])
            hit = None
            for st in stop_texts:
                i = new_text.find(st)
                if i != -1 and (hit is None or i < hit):
                    hit = i
            if hit is not None:
                return new_text[:hit]
        return self.tok.decode(ids[len(prompt_ids):])


def n_params(model):
    return int(sum(v.size for v in model.params.values()))


# ============================ корпус и данные ============================

def read_corpus():
    parts = []
    if os.path.isdir(CORPUS_DIR):
        for fn in sorted(os.listdir(CORPUS_DIR)):
            if fn.lower().endswith(".txt"):
                with open(os.path.join(CORPUS_DIR, fn), encoding="utf-8", errors="ignore") as f:
                    parts.append(f.read())
    return "\n".join(parts)


def read_corpus_files():
    parts = []
    if os.path.isdir(CORPUS_DIR):
        for fn in sorted(os.listdir(CORPUS_DIR)):
            if fn.lower().endswith(".txt"):
                with open(os.path.join(CORPUS_DIR, fn), encoding="utf-8", errors="ignore") as f:
                    parts.append((fn, f.read()))
    return parts


def build_train_val(model):
    """Честная контрольная: хвост ОСНОВНОГО корпуса БЕЗ диалогов.
    Диалоги в train подмешиваются с весом, но в val не подглядываются."""
    main_text = "\n".join(t for fn, t in read_corpus_files() if fn != "99_dialogs.txt")
    main_ids = np.array(model.tok.encode(main_text), dtype=np.int64)
    n_val = max(0, int(len(main_ids) * 0.05))
    if n_val > model.cfg["block_size"] + 2:
        val_ids, tr_main = main_ids[-n_val:], main_ids[:-n_val]
    else:
        val_ids, tr_main = None, main_ids
    train_ids = boost_with_dialogs(tr_main, dialog_ids(model.tok))
    return train_ids, val_ids


def get_batch(ids, B, T):
    ix = np.random.randint(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[i:i + T] for i in ix])
    y = np.stack([ids[i + 1:i + T + 1] for i in ix])
    return x, y


def dialog_text():
    path = os.path.join(CORPUS_DIR, "99_dialogs.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def dialog_ids(tok):
    txt = dialog_text()
    if len(txt) < 100:
        return None
    return np.array(tok.encode(txt), dtype=np.int64)


def boost_with_dialogs(main_ids, dids):
    """Диалоги держим ~20% смеси ПРИ ЛЮБОМ размере корпуса,
    чтобы личные уроки не тонули в миллионах символов книг."""
    if dids is None or len(dids) < 100:
        return main_ids
    target = int(len(main_ids) * 0.2)
    rep = int(np.clip(target // len(dids), 0, 400))
    if rep <= 0:
        return main_ids
    return np.concatenate([main_ids] + [dids] * rep)


def _corpus_hashes():
    hashes = set()
    if os.path.isdir(CORPUS_DIR):
        for fn in os.listdir(CORPUS_DIR):
            if fn.lower().endswith(".txt"):
                try:
                    with open(os.path.join(CORPUS_DIR, fn), "rb") as f:
                        hashes.add(hashlib.md5(f.read()).hexdigest())
                except Exception:
                    pass
    return hashes


def store_file(src, prefix="60_"):
    """Никогда не перезаписывает и не задваивает книги:
    - та же книга под ЛЮБЫМ именем (по содержимому) → пропуск;
    - новое содержимое под занятым именем → отдельный файл с отпечатком.
    ВАЖНО: одна книга = один файл. Если дописывать книги в конец одного
    text.txt и кормить заново, сохранится полный слепок — текст задвоится."""
    base = os.path.basename(src)
    stem, ext = os.path.splitext(base)
    with open(src, "rb") as f:
        data = f.read()
    h = hashlib.md5(data).hexdigest()
    if h in _corpus_hashes():
        return None, "dup"
    dst = os.path.join(CORPUS_DIR, prefix + base)
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if hashlib.md5(f.read()).hexdigest() == h:
                return dst, "dup"
        dst = os.path.join(CORPUS_DIR, f"{prefix}{stem}_{h[:6]}{ext}")
        if os.path.exists(dst):
            return dst, "dup"
    with open(dst, "wb") as f:
        f.write(data)
    return dst, "new"


def resize_params(params, old_vocab, new_vocab):
    o = {c: i for i, c in enumerate(old_vocab)}
    idx_map = np.array([o.get(c, -1) for c in new_vocab])
    have = idx_map >= 0
    rng = np.random.default_rng(0)
    old = params["tok_emb"]
    new_tok = (rng.standard_normal((len(new_vocab), old.shape[1])) * 0.02).astype(np.float32)
    new_tok[have] = old[idx_map[have]]
    params["tok_emb"] = new_tok
    old = params["Wout"]
    new_wout = (rng.standard_normal((old.shape[0], len(new_vocab))) * 0.02).astype(np.float32)
    new_wout[:, have] = old[:, idx_map[have]]
    params["Wout"] = new_wout
    old = params["bout"]
    new_b = np.zeros(len(new_vocab), np.float32)
    new_b[have] = old[idx_map[have]]
    params["bout"] = new_b
    return params


def ensure_vocab(model, text):
    if model.tok.kind == "char":
        vocab = sorted(set(text))
        if vocab == model.vocab:
            return vocab
        model.params = resize_params(model.params, model.vocab, vocab)
        model.vocab = vocab
        model.tok = Tok("char", vocab)
    else:
        have = set(model.vocab)
        add = [c for c in sorted(set(text)) if c not in have
               and (re.match(r"[A-Za-zА-Яа-яЁё0-9]", c) or c in " \n\t.,!?;:-—\"'()[]«»…")]
        if not add:
            return model.vocab
        vocab = model.vocab + add
        model.params = resize_params(model.params, model.vocab, vocab)
        model.vocab = vocab
        model.tok = Tok("bpe", vocab, model.tok.merges)
    model.opt = Adam(model.params, lr=model.cfg.get("lr", 1e-3),
                     wd=model.cfg.get("wd", 0.01))
    return model.vocab


# ============================ сохранение: атомарно + бэкапы ============================

def atomic_savez(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def backup_weights(tag="train"):
    """Ротируются только рабочие копии w_* (по mtime, последние 7).
    Архивные rebirth_old_* — бессрочные."""
    if not os.path.exists(WEIGHTS_PATH):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, f"w_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    try:
        shutil.copy2(WEIGHTS_PATH, dst)
        work = [f for f in os.listdir(BACKUP_DIR) if f.startswith("w_") and f.endswith(".npz")]
        work.sort(key=lambda f: os.path.getmtime(os.path.join(BACKUP_DIR, f)))
        for f in work[:-7]:
            os.remove(os.path.join(BACKUP_DIR, f))
    except Exception:
        pass


def save_model(model):
    atomic_savez(WEIGHTS_PATH, **model.params)
    cfg = dict(model.cfg)
    if model.tok.kind == "bpe":
        cfg["tokenizer"] = "bpe"
        atomic_write_json(TOK_PATH, {"merges": [list(m) for m in model.tok.merges]})
    meta = {"config": cfg, "vocab": model.vocab,
            "generation": getattr(model, "generation", 0)}
    born = None
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            born = json.load(f).get("born")
    meta["born"] = born or time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(META_PATH, meta)


def load_model():
    if not (os.path.exists(META_PATH) and os.path.exists(WEIGHTS_PATH)):
        return None
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    cfg = {**DEFAULT_CONFIG, **meta.get("config", {})}
    kind = cfg.get("tokenizer", "char")
    merges = None
    if kind == "bpe":
        vocab = list(meta["vocab"])
        if os.path.exists(TOK_PATH):
            with open(TOK_PATH, encoding="utf-8") as f:
                merges = [tuple(m) for m in json.load(f).get("merges", [])]
        else:
            print("  ⚠️ tokenizer.json не найден — кодирую посимвольно")
            merges = []
    else:
        text = read_corpus()
        vocab = sorted(set(text)) if text else meta["vocab"]
    params = dict(np.load(WEIGHTS_PATH))
    if vocab != meta["vocab"]:
        params = resize_params(params, meta["vocab"], vocab)
    model = MiniGPT(vocab, cfg, merges)
    model.params = params
    model.opt = Adam(params, lr=cfg.get("lr", 1e-3), wd=cfg.get("wd", 0.01))
    model.generation = meta.get("generation", 0)
    return model


def diary_append(entry):
    with open(DIARY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_diary():
    if not os.path.exists(DIARY_PATH):
        return []
    out = []
    with open(DIARY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def clip_grads(grads, max_norm):
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if total > max_norm and total > 0:
        s = max_norm / total
        for k in grads:
            grads[k] = grads[k] * s


# ============================ счётчики жизни ============================

def load_stats():
    base = {"total_steps": 0, "train_seconds": 0.0, "chats": 0, "messages": 0,
            "quiz_score": None, "quiz_pairs": 0, "growths": 0,
            "auto_grows": 0, "rollbacks": 0, "ladder_idx": 0,
            "best_val": None, "rebirths": 0, "cloud_runs": 0}
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, encoding="utf-8") as f:
            base.update(json.load(f))
    return base


def save_stats(stats):
    atomic_write_json(STATS_PATH, stats)


# ============================ ВЕЧНАЯ ПАМЯТЬ ============================

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"name": None, "city": None, "age": None, "facts": []}


def save_profile(p):
    os.makedirs(MEM_DIR, exist_ok=True)
    atomic_write_json(PROFILE_PATH, p)


def append_episode(user, harold_reply):
    os.makedirs(MEM_DIR, exist_ok=True)
    e = {"time": time.strftime("%Y-%m-%d %H:%M"),
         "user": user[:500], "harold": harold_reply[:500]}
    with open(EPISODES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")


def load_episodes():
    if not os.path.exists(EPISODES_PATH):
        return []
    out = []
    with open(EPISODES_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def count_episodes():
    return len(load_episodes())


FACT_PATTERNS = [
    ("name", re.compile(r"меня зовут\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{1,24})", re.I)),
    ("city", re.compile(r"я живу (?:в|во)\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\- ]{1,24})", re.I)),
    ("age", re.compile(r"мне\s+(\d{1,2})\s+лет", re.I)),
    ("likes", re.compile(r"(?:я люблю|мне нравится)\s+([^.!?\n]{2,50})", re.I)),
    ("work", re.compile(r"я работаю\s+([^.!?\n]{2,40})", re.I)),
]


def extract_facts(msg):
    p = load_profile()
    notes = []
    for key, rx in FACT_PATTERNS:
        m = rx.search(msg)
        if not m:
            continue
        val = m.group(1).strip().rstrip(",.; ")
        if not val:
            continue
        if key == "name":
            val = val[0].upper() + val[1:]
            if p.get("name") != val:
                p["name"] = val
                notes.append(f"имя хозяина: {val}")
        elif key == "city":
            if p.get("city") != val:
                p["city"] = val
                notes.append(f"живёт в {val}")
        elif key == "age":
            if p.get("age") != int(val):
                p["age"] = int(val)
                notes.append(f"возраст: {val}")
        else:
            fact = {"likes": "любит", "work": "работает: "}[key] + " " + val
            if fact.lower() not in [f.lower() for f in p["facts"]]:
                p["facts"].append(fact)
                notes.append(fact)
    if notes:
        save_profile(p)
    return p, notes


def memory_static_text(p):
    parts = []
    if p.get("name"):
        parts.append(f"Хозяина зовут {p['name']}.")
    if p.get("city"):
        parts.append(f"Хозяин живёт в {p['city']}.")
    if p.get("age"):
        parts.append(f"Хозяину {p['age']} лет.")
    for f in p.get("facts", [])[-8:]:
        parts.append(f"Хозяин {f}.")
    return " ".join(parts)


def trigrams(s):
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return set(s[i:i + 3] for i in range(max(len(s) - 2, 1)))


def recall_similar_episodes(query, k=3, thr=0.08):
    q = trigrams(query)
    if not q:
        return []
    scored = []
    for e in load_episodes():
        inter = len(q & trigrams(e["user"]))
        uni = len(q | trigrams(e["user"]))
        if uni:
            scored.append((inter / uni, e))
    scored.sort(key=lambda x: -x[0])
    return [e for s, e in scored[:k] if s > thr]


def build_chat_prompt(model, msg):
    T = model.cfg["block_size"]
    budget = max(24, T - 12)
    tail = model.tok.encode("\nПользователь: " + msg + "\nГарольд:")
    tip = None
    if len(tail) > budget:
        tail = tail[-budget:]
        tip = "сообщение длиннее контекста — начало срезано"
    rem = budget - len(tail)
    head = []
    if rem > 10:
        static = memory_static_text(load_profile())
        if static:
            sid = model.tok.encode("\n" + static)
            if len(sid) <= rem:
                head.extend(sid)
                rem -= len(sid)
        for e in recall_similar_episodes(msg, k=3):
            b = model.tok.encode(f"\nПользователь: {e['user']}\nГарольд: {e['harold']}")
            if len(b) <= rem:
                head.extend(b)
                rem -= len(b)
    if budget < 80:
        tip = tip or f"контекст мал ({T}) — для памяти в чате: python harold.py grow --ctx 384"
    return head + tail, tip


# ============================ уровни, XP, интеллект, трофеи ============================

LEVELS = [(0, "🐣 Новорождённый"), (150, "🍼 Младенец"), (500, "👶 Малыш"),
          (1200, "🧒 Почемучка"), (3000, "🎒 Школьник"), (7000, "🎓 Студент"),
          (14000, "🧑‍🎓 Выпускник"), (26000, "🔬 Аспирант"), (50000, "🧪 Учёный"),
          (90000, "🧠 Профессор"), (150000, "🦉 Мудрец"), (300000, "🌌 Легенда"),
          (600000, "👑 Властелин данных")]

GROWTH_LADDER = [("ctx", 384), ("mlp", 2), ("depth", 1),
                 ("ctx", 768), ("mlp", 2), ("depth", 1)]

MILESTONES = [
    ("first_train", "🐣 Первая тренировка", lambda c: c["stats"].get("total_steps", 0) > 0),
    ("gen5", "🧬 Пять поколений", lambda c: c["gens"] >= 5),
    ("gen25", "🧬 Двадцать пять поколений", lambda c: c["gens"] >= 25),
    ("gen100", "🧬 Сто поколений!", lambda c: c["gens"] >= 100),
    ("vocab30", "🔤 Словарь 30 токенов", lambda c: c["vocab"] >= 30),
    ("vocab60", "🔤 Словарь 60 токенов", lambda c: c["vocab"] >= 60),
    ("vocab100", "🔤 Словарь 100+ токенов", lambda c: c["vocab"] >= 100),
    ("bpe", "🗣 Освоил BPE — язык из частей слов",
     lambda c: c["arch"].get("tokenizer") == "bpe"),
    ("words200", "💭 Лексикон 200 слов", lambda c: c["words"] >= 200),
    ("words2000", "💭 Лексикон 2000 слов", lambda c: c["words"] >= 2000),
    ("corpus50k", "📚 Прочитано 50 000 символов", lambda c: c["corpus"] >= 50_000),
    ("corpus500k", "📚 Полмиллиона символов!", lambda c: c["corpus"] >= 500_000),
    ("corpus2m", "📚 Два миллиона символов!!", lambda c: c["corpus"] >= 2_000_000),
    ("corpus10m", "📚 ДЕСЯТЬ миллионов символов!!!", lambda c: c["corpus"] >= 10_000_000),
    ("scp50", "👽 50 статей Фонда SCP", lambda c: c.get("scp", 0) >= 50),
    ("scp250", "👽 250 статей Фонда SCP", lambda c: c.get("scp", 0) >= 250),
    ("loss15", "🎯 Ошибка ниже 1.5", lambda c: c["best_loss"] is not None and c["best_loss"] < 1.5),
    ("loss10", "🎯 Ошибка ниже 1.0", lambda c: c["best_loss"] is not None and c["best_loss"] < 1.0),
    ("loss07", "🎯 Ошибка ниже 0.7", lambda c: c["best_loss"] is not None and c["best_loss"] < 0.7),
    ("loss05", "🎯 Ошибка ниже 0.5", lambda c: c["best_loss"] is not None and c["best_loss"] < 0.5),
    ("dialogs20", "💬 Выучено 20 уроков", lambda c: c["dialogs"] >= 20),
    ("dialogs100", "💬 Выучено 100 уроков", lambda c: c["dialogs"] >= 100),
    ("name", "🫂 Запомнил имя хозяина", lambda c: bool(c["profile"].get("name"))),
    ("episodes50", "💭 50 воспоминаний", lambda c: c["episodes"] >= 50),
    ("quiz60", "🧠 Экзамен: вспомнил 60%", lambda c: (c["quiz"] or 0) >= 0.6),
    ("quiz85", "🧠 Экзамен: вспомнил 85%", lambda c: (c["quiz"] or 0) >= 0.85),
    ("vis95", "👁 Зрение 95%", lambda c: (c["vis"] or 0) >= 0.95),
    ("chat100", "☕ 100 сообщений в беседах", lambda c: c["messages"] >= 100),
    ("bigbrain", "🧠 Мозг на миллион параметров", lambda c: c["params"] >= 1_000_000),
    ("params5m", "🧠 Мозг на 5 000 000 параметров", lambda c: c["params"] >= 5_000_000),
    ("growth1", "🌱 Первый нейрогенез", lambda c: c["growths"] >= 1),
    ("auto1", "🚀 ПОТОЛОК ПРОБИТ: первый авто-рост", lambda c: c["auto_grows"] >= 1),
    ("rollback1", "↩️ Железная память: первый откат порчи", lambda c: c["rollbacks"] >= 1),
    ("rebirth", "🦋 Перерождение с сохранением души",
     lambda c: c["stats"].get("rebirths", 0) >= 1),
    ("cloud", "🌩 Облачное обучение (GitHub Actions)",
     lambda c: c["stats"].get("cloud_runs", 0) >= 1),
    ("browser", "🌐 Клон в браузере собран", lambda c: os.path.exists("site/index.html")),
    ("faces1", "👤 Узнаёт первое лицо", lambda c: c.get("faces", 0) >= 1),
    ("faces5", "👥 Узнаёт пять людей", lambda c: c.get("faces", 0) >= 5),
]


def load_milestones():
    if os.path.exists(MILES_PATH):
        with open(MILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_vision_acc():
    if os.path.exists(VISION_JSON):
        with open(VISION_JSON, encoding="utf-8") as f:
            return json.load(f).get("acc")
    return None


def gather_context():
    c = {}
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    text = read_corpus()
    c["corpus"] = len(text)
    c["words"] = len(set(re.findall(r"[а-яёa-z0-9]{2,}", text.lower()))) if text else 0
    c["vocab"] = len(meta.get("vocab", []))
    c["gens"] = meta.get("generation", 0)
    c["born"] = meta.get("born")
    c["arch"] = meta.get("config", {})
    c["dialogs"] = 0
    q = None
    path = os.path.join(CORPUS_DIR, "99_dialogs.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Пользователь:"):
                    q = 1
                elif line.startswith("Гарольд:") and q:
                    c["dialogs"] += 1
                    q = None
    c["wiki"] = c["scp"] = c["web"] = c["books"] = 0
    if os.path.isdir(CORPUS_DIR):
        for fn in os.listdir(CORPUS_DIR):
            if not fn.endswith(".txt"):
                continue
            fp = os.path.join(CORPUS_DIR, fn)
            if fn.startswith("50_wiki"):
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    c["wiki"] = sum(1 for line in f if line.startswith("Статья Википедии:"))
            elif "scp" in fn.lower():
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    c["scp"] += sum(1 for line in f if line.startswith("Статья SCP:"))
            elif fn.startswith(("40_web", "45_web")):
                c["web"] += 1
            elif fn.startswith(("60_", "61_")):
                c["books"] += 1
    c["files"] = sum(1 for fn in os.listdir(CORPUS_DIR)
                     if fn.endswith(".txt")) if os.path.isdir(CORPUS_DIR) else 0
    c["episodes"] = count_episodes()
    c["profile"] = load_profile()
    c["stats"] = st = load_stats()
    diary = read_diary()
    losses = [e["train_loss"] for e in diary if e.get("train_loss") is not None]
    vlosses = [e["val_loss"] for e in diary if e.get("val_loss") is not None]
    c["losses"], c["val_losses"] = losses, vlosses
    all_l = vlosses or losses
    c["best_loss"] = min(all_l) if all_l else None
    c["params"] = 0
    if os.path.exists(WEIGHTS_PATH):
        with np.load(WEIGHTS_PATH) as z:
            c["params"] = int(sum(a.size for a in z.values()))
    c["vis"] = load_vision_acc()
    c["faces"] = 0
    if os.path.exists(FACES_JSON):
        try:
            with open(FACES_JSON, encoding="utf-8") as f:
                c["faces"] = len(json.load(f).get("people", {}))
        except Exception:
            pass
    c["quiz"] = st.get("quiz_score")
    c["growths"] = st.get("growths", 0)
    c["auto_grows"] = st.get("auto_grows", 0)
    c["rollbacks"] = st.get("rollbacks", 0)
    c["ladder_idx"] = st.get("ladder_idx", 0)
    c["best_val"] = st.get("best_val")
    c["tok_kind"] = c["arch"].get("tokenizer", "char")
    c["merges"] = 0
    if os.path.exists(TOK_PATH):
        with open(TOK_PATH, encoding="utf-8") as f:
            c["merges"] = len(json.load(f).get("merges", []))
    c["milestones"] = load_milestones()
    return c


def compute_xp(c):
    s = c["stats"]
    return int(s.get("total_steps", 0) + c["gens"] * 80 + c["corpus"] // 250 +
               c["dialogs"] * 30 + c["episodes"] * 8 + c["words"] * 3 +
               c.get("scp", 0) * 12 + c.get("web", 0) * 6 + c.get("books", 0) * 15 +
               len(c["milestones"]) * 60 + s.get("growths", 0) * 100 +
               s.get("auto_grows", 0) * 150 + s.get("cloud_runs", 0) * 200)


def level_for(xp):
    idx = 0
    for i, (thr, _) in enumerate(LEVELS):
        if xp >= thr:
            idx = i
    cur_thr, title = LEVELS[idx]
    next_thr = LEVELS[idx + 1][0] if idx + 1 < len(LEVELS) else None
    return idx, title, cur_thr, next_thr


def compute_intellect(c):
    cl = lambda x: max(0.0, min(1.0, x))
    flu = cl((4.5 - c["best_loss"]) / 3.5) if c["best_loss"] is not None else 0.0
    mem = cl(c["quiz"] or 0)
    know = cl(c["corpus"] / 1_000_000)
    lex = cl(c["words"] / 5000)
    vis = cl(c["vis"] or 0)
    return int(round(100 * (0.30 * flu + 0.25 * mem + 0.20 * know + 0.15 * lex + 0.10 * vis)))


def intellect_label(x):
    if x < 10: return "💤 Спящий потенциал"
    if x < 25: return "🌱 Пробуждающийся"
    if x < 40: return "🔍 Любознательный"
    if x < 55: return "🧩 Смышлёный"
    if x < 70: return "💡 Ясный ум"
    if x < 85: return "🌟 Блестящий ум"
    return "🌌 Сияющий разум"


def check_milestones():
    c = gather_context()
    unlocked = c["milestones"]
    newly = []
    for mid, title, cond in MILESTONES:
        if mid not in unlocked:
            try:
                if cond(c):
                    unlocked[mid] = time.strftime("%Y-%m-%d %H:%M")
                    newly.append(title)
            except Exception:
                pass
    if newly:
        atomic_write_json(MILES_PATH, unlocked)
    return newly


def print_trophies(newly):
    for t in newly:
        print(f"\n  🏆 НОВЫЙ ТРОФЕЙ: {t}")
    print()


# ============================ рисовалки ============================

def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


def fmt_dur(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, _ = divmod(r, 60)
    return f"{h}ч {m}мин" if h else f"{m}мин"


def bar(frac, width=22, fill="█", empty="░"):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return fill * n + empty * (width - n)


def spark(vals, width=44):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return "·" * 6 + " (мало данных)"
    if len(vals) > width:
        idx = np.linspace(0, len(vals) - 1, width).astype(int)
        vals = [vals[i] for i in idx]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[int(7 * (v - lo) / (hi - lo))] for v in vals)


# ============================ обучение ============================

def lr_at(base, step, total, warm, floor=0.1):
    if step <= warm:
        return base * (0.3 + 0.7 * step / max(warm, 1))
    p = min(1.0, (step - warm) / max(1, total - warm))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def train_steps(model, ids, steps, batch_size, log_every=50, accum=1):
    T = model.cfg["block_size"]
    losses = []
    base_lr = model.opt.lr
    warm = min(20, steps)
    accum = max(1, accum)
    t0 = time.time()
    for step in range(1, steps + 1):
        model.opt.lr = lr_at(base_lr, step, steps, warm)
        grads = None
        for _ in range(accum):
            x, y = get_batch(ids, batch_size, T)
            logits, loss, cache = model.forward(x, y)
            g = model.backward(cache, logits, y)
            if grads is None:
                grads = {k: v * (1.0 / accum) for k, v in g.items()}
            else:
                for k in grads:
                    grads[k] += g[k] * (1.0 / accum)
            losses.append(loss)
        clip_grads(grads, model.cfg.get("clip", 1.0))
        model.opt.step(grads)
        if step % log_every == 0 or step == steps:
            print(f"    шаг {step:>5}/{steps}   ошибка {np.mean(losses[-log_every * accum:]):.3f}"
                  f"   lr {model.opt.lr:.2e}")
    model.opt.lr = base_lr
    return float(np.mean(losses)), time.time() - t0


def eval_loss(model, ids, iters=8, batch_size=4):
    T = model.cfg["block_size"]
    if ids is None or len(ids) < T + 2:
        return None
    vals = []
    for _ in range(iters):
        x, y = get_batch(ids, batch_size, T)
        _, loss, _ = model.forward(x, y)
        vals.append(loss)
    return float(np.mean(vals))


# ============================ интернет ============================
def _is_clean_text(t):
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 20:
        return False
    ok = sum(1 for c in letters if ("a" <= c <= "z") or ("A" <= c <= "Z")
             or ("а" <= c <= "я") or c in "Ёё")
    return ok / len(letters) >= 0.95

def fetch_wiki(lang, n):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    path = os.path.join(CORPUS_DIR, "50_wiki.txt")
    got = fails = 0
    while got < n and fails < 4:
        url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json"
               f"&generator=random&grnnamespace=0&grnlimit=20"
               f"&prop=extracts&exintro=1&explaintext=1&exlimit=max")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.load(r)
            pages = d.get("query", {}).get("pages", {}).values()
            for pg in pages:
                title, extract = pg.get("title", ""), (pg.get("extract") or "").strip()
                if len(extract) < 200 or not _is_clean_text(title + " " + extract):
                    continue
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"\n\nСтатья Википедии: {title}.\n{extract}\n")
                got += 1
                print(f"  + прочитал статью «{title}» ({len(extract)} символов)")
                if got >= n:
                    break
            fails = 0
            time.sleep(1.5)
        except urllib.error.HTTPError as e:
            fails += 1
            wait = int(e.headers.get("Retry-After", 0) or 0) or 5 * 2 ** fails
            print(f"  ! HTTP {e.code}, жду {wait} с (попытка {fails}/4)")
            time.sleep(wait)
        except KeyboardInterrupt:
            print("\n  прервано")
            break
        except Exception as e:
            fails += 1
            print("  ! не удалось скачать статью:", e)
            time.sleep(3)
    return got


def fetch_url_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        charset = (r.headers.get_content_charset() or "utf-8")
    page = raw.decode(charset, errors="ignore")
    page = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", page)
    txt = re.sub(r"(?s)<[^>]+>", " ", page)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    return txt.strip()


# ============================ CRAWL: SCP и списки ссылок ============================

SCP_SKIP_CLASSES = ("scp-image-block", "footnotes-footer", "page-rate-widget-box",
                    "creditRate", "licensebox", "page-tags", "footer-wikiwalk-nav")
SCP_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}


class ScpExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.skip = 0
        self.hidden = 0
        self.out = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div":
            if self.depth == 0:
                if a.get("id") == "page-content":
                    self.depth = 1
                return
            self.depth += 1
            if self.skip:
                self.skip += 1
            elif any(cl in (a.get("class") or "") for cl in SCP_SKIP_CLASSES):
                self.skip = 1
        if self.depth == 0:
            return
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in SCP_BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if self.depth == 0:
            return
        if tag in ("script", "style") and self.hidden:
            self.hidden -= 1
        if tag in SCP_BLOCK:
            self.out.append("\n")
        if tag == "div":
            self.depth -= 1
            if self.skip:
                self.skip -= 1

    def handle_data(self, data):
        if self.depth and not self.skip and not self.hidden:
            self.out.append(data)


def scp_clean(text):
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


def crawl_scp(base, start, end, suffix, delay, min_chars):
    try:
        if os.path.exists(META_PATH):
            with open(META_PATH, encoding="utf-8") as f:
                tok_kind = json.load(f).get("config", {}).get("tokenizer", "char")
            if tok_kind == "char" and "scp-ru" not in base:
                print("  ⚠️ токенизатор посимвольный: нерусские статьи раздуют словарь.")
                print("     Порядок: русские SCP → rebirth --bpe → потом английские.")
    except Exception:
        pass
    os.makedirs(CORPUS_DIR, exist_ok=True)
    path = os.path.join(CORPUS_DIR, "55_scp.txt")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = set(re.findall(r"^Статья SCP: (\S+?)\.$", f.read(), re.M))
        print(f"  уже прочитано раньше: {len(done)} статей — продолжаю с места остановки")
    saved = fails = 0
    for n in range(start, end + 1):
        slug = f"scp-{n:03d}{suffix}"
        name = slug.upper()
        if name in done:
            continue
        try:
            req = urllib.request.Request(
                f"{base}/{slug}",
                headers={
                    "User-Agent": UA,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Referer": base + "/",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                page = r.read().decode("utf-8", errors="ignore")
            ex = ScpExtractor()
            ex.feed(page)
            text = scp_clean("".join(ex.out))
            fails = 0
            if len(text) < min_chars:
                print(f"  - {name}: пропуск (мало текста: {len(text)})")
            else:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"\n\nСтатья SCP: {name}.\n{text[:30000]}\n")
                saved += 1
                print(f"  + {name} ({len(text)} символов)")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  - {name}: нет такой статьи")
            else:
                fails += 1
                wait = int(e.headers.get("Retry-After", 0) or 0) or 10 * 2 ** fails
                print(f"  ! HTTP {e.code}, жду {wait} с ({fails}/4)")
                if fails >= 4:
                    print("Слишком много ошибок, стоп. Позже продолжит с этого места.")
                    break
                time.sleep(wait)
        except KeyboardInterrupt:
            print("\nПрервано. Прогресс сохранён.")
            break
        except Exception as e:
            print(f"  ! {name}: {e}")
        time.sleep(delay)
    print(f"Готово: новых статей {saved}. Закрепить: python harold.py evolve --gens 3 --steps 600 --read 0 --lr 1e-3")
    print_trophies(check_milestones())


def crawl_urls(path, delay):
    if not os.path.exists(path):
        print("Файл со ссылками не найден")
        return
    os.makedirs(CORPUS_DIR, exist_ok=True)
    got = skip = 0
    with open(path, encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip() and not u.strip().startswith("#")]
    print(f"Ссылок в списке: {len(urls)}")
    for url in urls:
        h = hashlib.md5(url.encode()).hexdigest()[:8]
        dst = os.path.join(CORPUS_DIR, f"45_web_{h}.txt")
        if os.path.exists(dst):
            skip += 1
            continue
        try:
            txt = fetch_url_text(url)
            if len(txt) < 300:
                print(f"  - {url}: мало текста")
                continue
            with open(dst, "w", encoding="utf-8") as f:
                f.write(f"\n\nМатериал со страницы {url}.\n{txt}\n")
            got += 1
            print(f"  + {url} ({len(txt):,} символов)")
        except KeyboardInterrupt:
            print("\nПрервано. Прогресс сохранён.")
            break
        except Exception as e:
            print(f"  ! {url}: {e}")
        time.sleep(delay)
    print(f"Готово: новых страниц {got}, пропущено {skip}.")
    print("Закрепить: python harold.py evolve --gens 3 --steps 600 --read 0 --lr 1e-3")
    print_trophies(check_milestones())


# ============================ НЕЙРОГЕНЕЗ ============================

def grow_depth(model, n_new):
    """Новый слой = чистый лист: Wo=0 и W2=0 → оба выхода слоя дают ноль,
    функция мозга сохраняется ТОЧНО. Wq/Wk/Wv случайные: если и они нули,
    градиент замыкается (dWo∝Wv=0, dWv∝Wo=0) и внимание никогда не проснётся."""
    D = model.cfg["d_model"]
    f32 = np.float32
    rng = np.random.default_rng(1234)
    L = model.cfg["n_layers"]
    for i in range(n_new):
        pre = f"{L + i}."
        model.params[pre + "g1"] = np.ones(D, f32)
        model.params[pre + "g2"] = np.ones(D, f32)
        for nm in ("Wq", "Wk", "Wv"):
            model.params[pre + nm] = (rng.standard_normal((D, D)) / math.sqrt(D)).astype(f32)
        model.params[pre + "Wo"] = np.zeros((D, D), f32)
        model.params[pre + "W1"] = (rng.standard_normal((D, 4 * D)) / math.sqrt(D)).astype(f32)
        model.params[pre + "b1"] = np.zeros(4 * D, f32)
        model.params[pre + "W2"] = np.zeros((4 * D, D), f32)
        model.params[pre + "b2"] = np.zeros(D, f32)
    model.cfg["n_layers"] = L + n_new
    model.L = L + n_new


def grow_mlp(model, factor):
    D = model.cfg["d_model"]
    f32 = np.float32
    rng = np.random.default_rng(5678)
    for l in range(model.cfg["n_layers"]):
        pre = f"{l}."
        W1, W2, b1 = model.params[pre + "W1"], model.params[pre + "W2"], model.params[pre + "b1"]
        H = W1.shape[1]
        new_W1 = np.zeros((D, H * factor), f32)
        new_W1[:, :H] = W1
        new_W1[:, H:] = (rng.standard_normal((D, H * (factor - 1))) * 0.02).astype(f32)
        new_W2 = np.zeros((H * factor, D), f32)
        new_W2[:H] = W2
        new_b1 = np.zeros(H * factor, f32)
        new_b1[:H] = b1
        model.params[pre + "W1"], model.params[pre + "W2"], model.params[pre + "b1"] = \
            new_W1, new_W2, new_b1


def grow_ctx(model, new_T):
    old = model.params["pos_emb"]
    T0, D = old.shape
    new = np.zeros((new_T, D), np.float32)
    new[:T0] = old
    new[T0:] = old[-1]
    model.params["pos_emb"] = new
    model.cfg["block_size"] = new_T
    model.mask = None


def growth_step(model, st):
    idx = st.get("ladder_idx", 0)
    kind, val = None, None
    for _ in range(len(GROWTH_LADDER)):
        kind, val = GROWTH_LADDER[idx % len(GROWTH_LADDER)]
        idx += 1
        if not (kind == "ctx" and val <= model.cfg["block_size"]):
            break
    st["ladder_idx"] = idx
    return kind, val


def apply_growth(model, kind, val):
    if kind == "ctx":
        if model.cfg["block_size"] >= 1536:
            return None
        grow_ctx(model, val)
        desc = f"контекст → {val} токенов"
    elif kind == "mlp":
        if model.params["0.W1"].shape[1] >= 8192:
            return None
        grow_mlp(model, 2)
        desc = f"мышление ×2 (скрытых нейронов: {model.params['0.W1'].shape[1]})"
    elif kind == "depth":
        if model.cfg["n_layers"] >= 12:
            return None
        grow_depth(model, 1)
        desc = f"глубина → {model.cfg['n_layers']} слоёв"
    else:
        return None
    model.cfg["lr"] = min(model.cfg.get("lr", 1e-3) * 1.25, 2.5e-3)
    model.opt = Adam(model.params, lr=model.cfg["lr"], wd=model.cfg.get("wd", 0.01))
    return desc


def shapes_match(best_path, shapes):
    if not os.path.exists(best_path):
        return False
    try:
        with np.load(best_path) as z:
            return (set(z.files) == set(shapes) and
                    all(z[k].shape == s for k, s in shapes.items()))
    except Exception:
        return False


# ============================ парсинг диалогов ============================

def parse_dialog_pairs(limit=None):
    pairs, q = [], None
    for line in dialog_text().splitlines():
        line = line.strip()
        if line.startswith("Пользователь:"):
            q = line[len("Пользователь:"):].strip()
        elif line.startswith("Гарольд:") and q:
            pairs.append((q, line[len("Гарольд:"):].strip()))
            q = None
    return pairs[:limit] if limit else pairs


def norm_text(s):
    s = re.sub(r"[^\wа-яё ]+", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


# ============================ команды: жизнь ============================

def make_brain(text, args_like):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"d_model": args_like.d, "n_layers": args_like.layers,
                "n_heads": args_like.heads, "block_size": args_like.ctx})
    merges = None
    if getattr(args_like, "bpe", False) and len(text) >= 50_000:
        n_m = min(getattr(args_like, "merges", 2000), max(500, len(text) // 300))
        print(f"🧬 Обучаю BPE-язык ({n_m} слияний) — это один раз и навсегда...")
        merges = train_bpe(text, n_merges=n_m)
        vocab = bpe_vocab_from_merges(text, merges)
        cfg["tokenizer"] = "bpe"
        print(f"   язык готов: {len(vocab)} токенов (вместо {len(set(text))} символов)")
    else:
        if getattr(args_like, "bpe", False):
            print("  ⚠️ корпус < 50 000 символов — BPE пока рано, беру символьный режим")
        vocab = sorted(set(text))
    return MiniGPT(vocab, cfg, merges), cfg


def cmd_init(args):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(MEM_DIR, exist_ok=True)
    seed = os.path.join(CORPUS_DIR, "00_start.txt")
    if not os.path.exists(seed):
        with open(seed, "w", encoding="utf-8") as f:
            f.write(SEED_CORPUS)
    if os.path.exists(META_PATH):
        print("Гарольд уже существует. Новый больший мозг: python harold.py rebirth --bpe")
        return
    text = read_corpus()
    model, cfg = make_brain(text, args)
    model.generation = 0
    save_model(model)
    print(f"Мистер Гарольд создан! Параметров мозга: {n_params(model):,}, "
          f"токенизатор: {cfg.get('tokenizer', 'char')}, словарь: {len(model.vocab)}.")
    print("\nЧто дальше:")
    print("  python harold.py crawl scp --start 1 --end 500   # Фонд SCP")
    print("  python harold.py evolve --gens 5 --lr 1e-3 --auto-grow")
    print("  python harold.py chat")
    print("  python harold.py face-add Арсен фото.jpg         # познакомить в лицо")


def cmd_rebirth(args):
    if not (os.path.exists(META_PATH) or os.path.exists(WEIGHTS_PATH)):
        print("Старого Гарольда нет — просто создайте: python harold.py init")
        return
    text = read_corpus()
    if len(text) < 500:
        print("Корпус пуст — перерождаться не из чего. Сначала: crawl / read / feed / teach")
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if os.path.exists(WEIGHTS_PATH):
        shutil.copy2(WEIGHTS_PATH, os.path.join(BACKUP_DIR, f"rebirth_old_{ts}.npz"))
    if os.path.exists(META_PATH):
        shutil.copy2(META_PATH, os.path.join(BACKUP_DIR, f"rebirth_old_{ts}.json"))
    model, cfg = make_brain(text, args)
    model.generation = 0
    save_model(model)
    st = load_stats()
    st["rebirths"] = st.get("rebirths", 0) + 1
    st["best_val"] = None
    st["ladder_idx"] = 0
    save_stats(st)
    if os.path.exists(BEST_PATH):
        os.remove(BEST_PATH)
    print(f"🦋 ПЕРЕРОЖДЕНИЕ! Новый мозг: {n_params(model):,} параметров · "
          f"токенизатор {cfg.get('tokenizer', 'char')} · контекст {cfg['block_size']}")
    print("   Корпус, вечная память о тебе, фотоальбом, дневник и трофеи — НЕ тронуты.")
    print("   Переучись: python harold.py evolve --gens 10 --steps 800 --lr 1e-3 --auto-grow")


def run_generation(model, gen, steps, batch_size, online_read, lang, accum=1):
    if online_read > 0:
        got = fetch_wiki(lang, online_read)
        if got:
            print(f"  новых статей прочитано: {got}")
    text = read_corpus()
    if len(text) < 500:
        print("  Корпус слишком мал: crawl scp / read / learn / ingest / feed / teach")
        return None
    model.opt.lr = model.cfg.get("lr", 1e-3)
    vocab = ensure_vocab(model, text)
    train_ids, val_ids = build_train_val(model)
    print(f"  обучение: {steps} шагов × accum {accum} на {len(train_ids):,} токенах...")
    tr_loss, dt = train_steps(model, train_ids, steps, batch_size, accum=accum)
    v_loss = eval_loss(model, val_ids)
    model.cfg["lr"] = max(model.cfg.get("lr", 1e-3) * 0.985, 2e-4)
    model.opt.lr = model.cfg["lr"]
    sample = model.generate(
        model.tok.encode("\nПользователь: привет\nГарольд:"),
        120, temp=model.cfg.get("temperature", 0.8),
        top_p=0.92, rep_pen=1.05,
        stop_texts=["\nПользователь:", "\n\n"]).strip()
    model.generation = gen
    save_model(model)
    st = load_stats()
    st["total_steps"] += steps * accum
    st["train_seconds"] += dt
    save_stats(st)
    ppl = math.exp(min(tr_loss, 20))
    entry = {
        "generation": gen, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_loss": round(tr_loss, 4),
        "val_loss": round(v_loss, 4) if v_loss is not None else None,
        "ppl": round(ppl, 2), "steps": steps, "accum": accum,
        "lr": round(model.cfg["lr"], 6), "ctx": model.cfg["block_size"],
        "corpus_chars": len(text), "vocab_size": len(vocab),
        "params": n_params(model),
        "sample": sample[:200].replace("\n", " "),
    }
    diary_append(entry)
    ctrl = f"  контрольная: {v_loss:.3f}" if v_loss is not None else ""
    print(f"  ошибка: {tr_loss:.3f}{ctrl}   перплексия: {ppl:.1f}")
    print(f"  словарь: {len(vocab)}   параметры: {n_params(model):,}   "
          f"корпус: {len(text):,} симв.   контекст: {model.cfg['block_size']}")
    print(f"  Гарольд говорит: «{entry['sample'][:150]}»")
    fill = len(text) / max(n_params(model) * 18, 1)
    if fill > 1:
        print("  💡 Знаний стало много для этого мозга — потолок близко.")
        print("     Включи авторост: evolve --auto-grow  (или вручную: grow --mlp 2)")
    return entry


def cmd_evolve(args):
    model = load_model()
    if model is None:
        print("Сначала создайте Гарольда: python harold.py init")
        return
    cur_lr = model.cfg.get("lr", 1e-3)
    if args.lr:
        model.cfg["lr"] = args.lr
        model.opt.lr = args.lr
        save_model(model)
        print(f"  темп обучения зафиксирован: lr = {args.lr}")
    elif cur_lr < 3.5e-4:
        print(f"  ⚠️ темп обучения сильно снижен прошлыми поколениями (lr={cur_lr:.1e}).")
        print("     Рекомендую: python harold.py evolve --lr 1e-3")
    backup_weights("evolve")
    st = load_stats()
    best = st.get("best_val")
    bad = 0
    for _ in range(args.gens):
        gen = model.generation + 1
        print(f"\n———— Поколение {gen} " + "—" * 30)
        pre_shapes = {k: v.shape for k, v in model.params.items()}
        entry = run_generation(model, gen, args.steps, args.batch,
                               args.read, args.lang, accum=args.accum)
        if entry is None:
            break
        val = entry["val_loss"]
        score = val if val is not None else entry["train_loss"]
        if val is not None:
            if best is None or score < best - 0.003:
                best = score
                st["best_val"] = round(best, 4)
                try:
                    shutil.copyfile(WEIGHTS_PATH, BEST_PATH)
                except Exception:
                    pass
                bad = 0
                print(f"  📈 Новый рекорд понимания: {best:.3f}")
            else:
                bad += 1
                if score > best * 1.12 and shapes_match(BEST_PATH, pre_shapes):
                    with np.load(BEST_PATH) as z:
                        model.params = {k: z[k] for k in z.files}
                    model.cfg["lr"] = max(model.cfg.get("lr", 1e-3) * 0.5, 5e-5)
                    model.opt = Adam(model.params, lr=model.cfg["lr"],
                                     wd=model.cfg.get("wd", 0.01))
                    st["rollbacks"] = st.get("rollbacks", 0) + 1
                    save_model(model)
                    diary_append({"event": "откат к лучшей версии мозга (защита знаний)",
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                    print("  ↩️ Знания портились — откат к лучшей версии, темп снижен")
                else:
                    print(f"  (без рекорда {bad}/{args.patience})")
            save_stats(st)
        if args.auto_grow:
            fill = len(read_corpus()) / max(n_params(model) * 18, 1)
            if bad >= args.patience or fill > 1.25:
                kind, gv = growth_step(model, st)
                desc = apply_growth(model, kind, gv)
                if desc:
                    st["growths"] = st.get("growths", 0) + 1
                    st["auto_grows"] = st.get("auto_grows", 0) + 1
                    print(f"  🚀 ПОТОЛОК ПРОБИТ: {desc}")
                    print("     старые знания целы (нулевая инициализация).")
                    diary_append({"event": f"авто-рост: {desc}",
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                else:
                    print("  ⚙️ Ступень недоступна (предел архитектуры) — попробует следующую.")
                bad = 0
                best = None
                st["best_val"] = None
                save_stats(st)
                save_model(model)
        if args.goal is not None:
            s = entry["val_loss"] if entry["val_loss"] is not None else entry["train_loss"]
            if s <= args.goal:
                print(f"\n  🎯 Цель достигнута: ошибка {s:.3f} ≤ {args.goal}.")
                break
    print_trophies(check_milestones())
    print(f"Текущее поколение: {model.generation}. Дневник: python harold.py diary --chart")


def cmd_train(args):
    model = load_model()
    if model is None:
        print("Сначала создайте Гарольда: python harold.py init")
        return
    text = read_corpus()
    if len(text) < 500:
        print("Корпус слишком мал. Сначала: crawl scp / read / learn / ingest / feed / teach")
        return
    backup_weights("train")
    ensure_vocab(model, text)
    if args.lr:
        model.cfg["lr"] = args.lr
        model.opt.lr = args.lr
    ids, _ = build_train_val(model)
    print(f"Обучаю Гарольда ({args.steps} шагов, виртуальный батч {args.batch * args.accum})...")
    loss, dt = train_steps(model, ids, args.steps, args.batch, accum=args.accum)
    save_model(model)
    st = load_stats()
    st["total_steps"] += args.steps * args.accum
    st["train_seconds"] += dt
    save_stats(st)
    print(f"Готово за {dt:.0f} с, ошибка {loss:.3f}. Поговорите: python harold.py chat")
    print_trophies(check_milestones())


# ============================ команды: питание знаниями ============================

def cmd_crawl(args):
    if args.source == "scp":
        crawl_scp(args.base, args.start, args.end, args.suffix, args.delay, args.min_chars)
    else:
        crawl_urls(args.file, args.delay)


def cmd_read(args):
    got = fetch_wiki(args.lang, args.n)
    print(f"Готово: прочитано {got} статей. Закрепить: python harold.py train --lr 1e-3")
    print_trophies(check_milestones())


def cmd_learn(args):
    try:
        txt = fetch_url_text(args.url)
    except Exception as e:
        print("Не удалось прочитать страницу:", e)
        return
    if len(txt) < 300:
        print("На странице слишком мало текста — Гарольд остался голодным.")
        return
    os.makedirs(CORPUS_DIR, exist_ok=True)
    h = hashlib.md5(args.url.encode()).hexdigest()[:8]
    dst = os.path.join(CORPUS_DIR, f"40_web_{h}.txt")
    if os.path.exists(dst):
        print("Эту страницу Гарольд уже читал. Запустите: python harold.py train")
        return
    with open(dst, "w", encoding="utf-8") as f:
        f.write(f"\n\nМатериал со страницы {args.url}.\n{txt}\n")
    print(f"Прочитал страницу ({len(txt):,} символов). Закрепить: python harold.py train --lr 1e-3")
    print_trophies(check_milestones())


def cmd_ingest(args):
    if not os.path.exists(args.file):
        print("Файл не найден")
        return
    os.makedirs(CORPUS_DIR, exist_ok=True)
    dst, status = store_file(args.file, prefix="60_")
    if status == "dup":
        print("Эта книга уже в знаниях (то же содержимое) — пропускаю, ничего не затёрто.")
    else:
        print(f"Книга добавлена: {os.path.basename(dst)} ({os.path.getsize(dst):,} байт)")
        print("Закрепить: python harold.py train --steps 800 --lr 1e-3")
    print_trophies(check_milestones())


def cmd_feed(args):
    src = args.path
    if os.path.isfile(src):
        cmd_ingest(argparse.Namespace(file=src))
        return
    if not os.path.isdir(src):
        print("Путь не найден (нужна папка или файл)")
        return
    os.makedirs(CORPUS_DIR, exist_ok=True)
    n = total = dup = 0
    for root, _, files in os.walk(src):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            fp = os.path.join(root, fn)
            dst, status = store_file(fp, prefix=f"61_{n:03d}_")
            if status == "new":
                total += os.path.getsize(dst)
                n += 1
            else:
                dup += 1
    print(f"Добавлено книг: {n} ({total:,} байт), пропущено дубликатов: {dup}. Большой пир!")
    print("Теперь длинный обед: python harold.py evolve --gens 5 --steps 800 --lr 1e-3 --auto-grow")
    print_trophies(check_milestones())


def cmd_teach(args):
    if not os.path.exists(META_PATH):
        print("Сначала: python harold.py init")
        return
    os.makedirs(CORPUS_DIR, exist_ok=True)
    path = os.path.join(CORPUS_DIR, "99_dialogs.txt")
    print("Учим Гарольда. Пустой вопрос — выход.")
    print("Короткие пары запоминаются лучше всего. После урока: train --steps 400 --lr 1e-3")
    while True:
        try:
            q = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        a = input("Ответ Гарольда: ").strip()
        if not a:
            print("  пустой ответ — пропускаю")
            continue
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\nПользователь: {q}\nГарольд: {a}\n")
        print("  запомнил!")
    print_trophies(check_milestones())


# ============================ команды: общение ============================

def cmd_chat(args):
    model = load_model()
    if model is None:
        print("Сначала: python harold.py init (и обучите его!)")
        return
    text = read_corpus()
    if text:
        ensure_vocab(model, text)
    p = load_profile()
    known = 0
    if os.path.exists(FACES_JSON):
        try:
            with open(FACES_JSON, encoding="utf-8") as f:
                known = len(json.load(f).get("people", {}))
        except Exception:
            pass
    print("=" * 62)
    print("Мистер Гарольд на связи. Пустая строка или /exit — выход.")
    print(f"поколение {model.generation} · воспоминаний: {count_episodes()}"
          + (f" · знает в лицо: {known}" if known else "")
          + (f" · хозяин: {p['name']}" if p.get("name") else ""))
    print("Команды: /save — записать обмен как урок, /memory — что он помнит")
    print("=" * 62)
    st = load_stats()
    st["chats"] = st.get("chats", 0) + 1
    save_stats(st)
    last_q = last_a = None
    tip_shown = False
    while True:
        try:
            msg = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg or msg == "/exit":
            break
        if msg == "/memory":
            p = load_profile()
            print("  💭 Память:", memory_static_text(p) or "(пока пуста)")
            for e in load_episodes()[-3:]:
                print(f"  [{e['time']}] Вы: {e['user'][:60]} | Он: {e['harold'][:60]}")
            continue
        if msg == "/save":
            if last_q:
                with open(os.path.join(CORPUS_DIR, "99_dialogs.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\nПользователь: {last_q}\nГарольд: {last_a}\n")
                print("  📖 Урок записан! Закрепите: python harold.py train --steps 400 --lr 1e-3")
            else:
                print("  Пока нечего сохранять.")
            continue
        _, notes = extract_facts(msg)
        for note in notes:
            print(f"  [🧠 навсегда запомнил: {note}]")
        ids, tip = build_chat_prompt(model, msg)
        if tip and not tip_shown:
            print(f"  (подсказка: {tip})")
            tip_shown = True
        reply = model.generate(ids, args.max_new, temp=args.temp, top_k=args.topk,
                               top_p=args.topp, rep_pen=args.reppen,
                               stop_texts=["\nПользователь:", "\n\n"]).strip()
        print("Гарольд:", reply or "(пока молчит — обучите его ещё: harold evolve --lr 1e-3)")
        append_episode(msg, reply or "…")
        last_q, last_a = msg, reply
        st = load_stats()
        st["messages"] = st.get("messages", 0) + 1
        save_stats(st)
    print_trophies(check_milestones())
    print("До встречи! Разговор сохранён в вечную память. 💾")


def cmd_say(args):
    model = load_model()
    if model is None:
        print("Сначала: python harold.py init")
        return
    text = read_corpus()
    if text:
        ensure_vocab(model, text)
    ids = model.tok.encode(args.prompt)
    print(model.generate(ids, args.max_new, temp=args.temp, top_k=args.topk,
                         top_p=args.topp, rep_pen=args.reppen))


def cmd_think(args):
    model = load_model()
    if model is None:
        print("Сначала: python harold.py init")
        return
    text = read_corpus()
    if text:
        ensure_vocab(model, text)
    seed = args.prompt if args.prompt else "Мысли Гарольда:"
    ids = model.tok.encode(seed)
    print("💭 " + model.generate(ids, args.max_new, temp=1.05, top_p=0.95, rep_pen=1.1))


def cmd_remember(args):
    if not os.path.exists(META_PATH):
        print("Сначала: python harold.py init")
        return
    text = " ".join(args.text).strip()
    if not text:
        print("Что запомнить? Например: python harold.py remember \"Меня зовут Арсен\"")
        return
    _, notes = extract_facts(text)
    if not notes:
        p = load_profile()
        fact = text[0].lower() + text[1:] if not text[0].isupper() else text
        if fact.lower() not in [f.lower() for f in p["facts"]]:
            p["facts"].append(fact)
            save_profile(p)
            notes = [fact]
    for n in notes:
        print(f"🧠 Навсегда запомнил: {n}")
    print("Проверить: python harold.py memory")
    print_trophies(check_milestones())


def cmd_memory(args):
    if not os.path.exists(META_PATH):
        print("Сначала: python harold.py init")
        return
    p = load_profile()
    eps = load_episodes()
    print("———— 💭 Вечная память Гарольда ————————")
    print(" О хозяине:", memory_static_text(p) or "(пока ничего)")
    print(f" Воспоминаний о разговорах: {len(eps)}")
    for e in eps[-args.show:]:
        print(f"  [{e['time']}] Вы: {e['user'][:70]}")
        print(f"             Он: {e['harold'][:70]}")


def cmd_forget(args):
    if args.all or args.facts:
        p = load_profile()
        if args.all:
            p = {"name": None, "city": None, "age": None, "facts": []}
        else:
            p["facts"] = []
        save_profile(p)
        print("Факты стёрты.")
    if args.all or args.episodes:
        if os.path.exists(EPISODES_PATH):
            os.remove(EPISODES_PATH)
        print("Воспоминания о разговорах стёрты.")
    if args.name and not args.all:
        p = load_profile()
        p["name"] = None
        save_profile(p)
        print("Имя забыто.")
    if not (args.all or args.facts or args.episodes or args.name):
        print("Укажите что забыть: --facts --episodes --name или --all")


# ============================ ЭКЗАМЕН ============================

def cmd_quiz(args):
    model = load_model()
    if model is None:
        print("Сначала: python harold.py init")
        return
    text = read_corpus()
    if text:
        ensure_vocab(model, text)
    pairs = parse_dialog_pairs(limit=args.n or None)
    if not pairs:
        print("Уроков ещё нет: python harold.py teach (или /save в чате)")
        return
    print(f"———— 📝 Экзамен для Гарольда: {len(pairs)} уроков ————————")
    ratios = []
    for q, a in pairs:
        ids = model.tok.encode(f"\nПользователь: {q}\nГарольд:")
        gen = model.generate(ids, len(a) + 25, temp=0.5, top_k=5,
                             stop_texts=["\n"]).strip()
        r = difflib.SequenceMatcher(None, norm_text(gen), norm_text(a)).ratio()
        ratios.append(r)
        mark = "✓" if r >= 0.6 else ("~" if r >= 0.3 else "✗")
        print(f"  {mark} В: {q[:44]:<44} → «{gen[:40]}» ({r:.0%})")
    score = float(np.mean(ratios))
    st = load_stats()
    st["quiz_score"] = round(score, 3)
    st["quiz_pairs"] = len(pairs)
    save_stats(st)
    verdict = ("🏆 Блестящая память!" if score >= 0.85 else
               "😊 Хорошо помнит уроки" if score >= 0.6 else
               "📖 Помнит частично — стоит потренировать" if score >= 0.3 else
               "💤 Пока помнит слабо — больше train --lr 1e-3 и evolve")
    print(f"\nИтог: вспомнил {score:.0%} уроков. {verdict}")
    print_trophies(check_milestones())


# ============================ ручной нейрогенез ============================

def cmd_grow(args):
    model = load_model()
    if model is None:
        print("Сначала: python harold.py init")
        return
    if not (args.depth or args.mlp or args.ctx):
        print("Нейрогенез — рост мозга без потери знаний:")
        print("  grow --depth 2     +2 слоя (сначала молчат, потом учатся)")
        print("  grow --mlp 2       мышление в 2 раза шире")
        print("  grow --ctx 384     длиннее память на один взгляд (важно для чата!)")
        print("Либо доверься лестнице: evolve --auto-grow")
        return
    before = n_params(model)
    backup_weights("grow")
    st = load_stats()
    did = []
    if args.ctx and args.ctx > model.cfg["block_size"]:
        grow_ctx(model, args.ctx)
        did.append(f"контекст {model.cfg['block_size']}")
    if args.mlp and args.mlp > 1:
        grow_mlp(model, args.mlp)
        did.append(f"MLP ×{args.mlp}")
    if args.depth and args.depth > 0:
        grow_depth(model, args.depth)
        did.append(f"+{args.depth} слоя(ёв)")
    if not did:
        print("Нечего растить: проверьте аргументы")
        return
    model.cfg["lr"] = min(model.cfg.get("lr", 1e-3) * 1.25, 2.5e-3)
    model.opt = Adam(model.params, lr=model.cfg["lr"], wd=model.cfg.get("wd", 0.01))
    st["growths"] = st.get("growths", 0) + 1
    save_stats(st)
    save_model(model)
    diary_append({"event": "нейрогенез: " + ", ".join(did),
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("🌱 НЕЙРОГЕНЕЗ ЗАВЕРШЁН: " + ", ".join(did))
    print(f"   мозг: {before:,} → {n_params(model):,} параметров (+{n_params(model) - before:,})")
    print("   все прошлые знания не тронуты. Раскачать: python harold.py train --steps 800 --lr 1e-3")
    print_trophies(check_milestones())


# ============================ зрение: цифры ============================

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def ensure_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)
    for fn in MNIST_FILES:
        path = os.path.join(MNIST_DIR, fn)
        if not os.path.exists(path):
            print(f"  скачиваю {fn} ...")
            urllib.request.urlretrieve(MNIST_URL + fn, path)


def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[:4], "big")
    n = int.from_bytes(data[4:8], "big")
    if magic == 2051:
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28)
    return np.frombuffer(data, np.uint8, offset=8)


def cmd_vision_train(args):
    try:
        ensure_mnist()
    except Exception as e:
        print("Не удалось скачать MNIST:", e)
        return
    Xtr = load_idx(os.path.join(MNIST_DIR, MNIST_FILES[0])).reshape(-1, 784).astype(np.float32) / 255.0
    Ytr = load_idx(os.path.join(MNIST_DIR, MNIST_FILES[1])).astype(np.int64)
    Xte = load_idx(os.path.join(MNIST_DIR, MNIST_FILES[2])).reshape(-1, 784).astype(np.float32) / 255.0
    Yte = load_idx(os.path.join(MNIST_DIR, MNIST_FILES[3])).astype(np.int64)
    rng = np.random.default_rng(7)
    if os.path.exists(VISION_PATH):
        params = dict(np.load(VISION_PATH))
        print("Продолжаю обучение зрения Гарольда...")
    else:
        params = {
            "W1": (rng.standard_normal((784, 256)) / math.sqrt(784)).astype(np.float32),
            "b1": np.zeros(256, np.float32),
            "W2": (rng.standard_normal((256, 10)) / math.sqrt(256)).astype(np.float32),
            "b2": np.zeros(10, np.float32),
        }
        print("Гарольд открывает глаза и учится видеть (цифры MNIST)...")
    opt = Adam(params, lr=1e-3)
    n, bs = len(Xtr), 64
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(n)
        losses = []
        for i in range(0, n - bs, bs):
            xb, yb = Xtr[order[i:i + bs]], Ytr[order[i:i + bs]]
            h1 = np.maximum(xb @ params["W1"] + params["b1"], 0)
            logits = h1 @ params["W2"] + params["b2"]
            loss, probs = ce_loss(logits, yb)
            dlog = probs.copy()
            dlog[np.arange(bs), yb] -= 1.0
            dlog /= bs
            dh1 = (dlog @ params["W2"].T) * (h1 > 0)
            opt.step({"W1": xb.T @ dh1, "b1": dh1.sum(0),
                      "W2": h1.T @ dlog, "b2": dlog.sum(0)})
            losses.append(loss)
        te_n = min(len(Xte), 2000)
        h1 = np.maximum(Xte[:te_n] @ params["W1"] + params["b1"], 0)
        acc = float(((h1 @ params["W2"] + params["b2"]).argmax(1) == Yte[:te_n]).mean())
        print(f"  эпоха {ep}: ошибка {np.mean(losses):.3f}, точность на тесте: {acc * 100:.1f}%")
    np.savez(VISION_PATH, **params)
    h1 = np.maximum(Xte @ params["W1"] + params["b1"], 0)
    full_acc = float(((h1 @ params["W2"] + params["b2"]).argmax(1) == Yte).mean())
    atomic_write_json(VISION_JSON, {"acc": round(full_acc, 4)})
    print(f"Гарольд научился видеть цифры! Итог: {full_acc * 100:.2f}% на всех 10 000 тестах.")
    print_trophies(check_milestones())


def cmd_see(args):
    if not os.path.exists(VISION_PATH):
        print("Сначала обучите зрение: python harold.py vision-train")
        return
    try:
        from PIL import Image
    except ImportError:
        print("Нужна библиотека Pillow: pip install pillow")
        return
    img = Image.open(args.image).convert("L").resize((28, 28))
    x = np.asarray(img, dtype=np.float32) / 255.0
    if x.mean() > 0.5:
        x = 1.0 - x
    params = dict(np.load(VISION_PATH))
    h1 = np.maximum(x.reshape(1, 784) @ params["W1"] + params["b1"], 0)
    pr = softmax((h1 @ params["W2"] + params["b2"])[0].astype(np.float64))
    pred = int(pr.argmax())
    print(f"Гарольд видит цифру: {pred}  (уверенность {pr[pred] * 100:.0f}%)")
    print("  вероятности:", " ".join(f"{i}:{p:.2f}" for i, p in enumerate(pr)))


# ============================ ЛИЦА: Гарольд узнаёт людей ============================

FACE_SIZE = 64          # каждое лицо сжимается в квадрат 64x64
PCA_DIM = 48            # длина «отпечатка лица»
FACE_THR = 0.45         # ниже этой похожести — «не знаю»


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        print("Нужен OpenCV (бесплатно): pip install opencv-python")
        return None


def load_face_gallery():
    if os.path.exists(FACES_JSON):
        with open(FACES_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"people": {}, "next_id": 0}


def save_face_gallery(g):
    os.makedirs(FACES_DIR, exist_ok=True)
    atomic_write_json(FACES_JSON, g)


def _load_gallery_npz():
    if os.path.exists(FACES_NPZ):
        with np.load(FACES_NPZ) as z:
            return list(z["X"]), list(z["ids"])
    return [], []


def _save_gallery_npz(X, ids):
    """Пересобирает фотоальбом и «зрение на лица» (собственные лица, PCA)."""
    if not X:
        if os.path.exists(FACES_NPZ):
            os.remove(FACES_NPZ)
        return
    Xa = np.asarray(X, dtype=np.float32)
    mean = Xa.mean(axis=0)
    Xc = Xa - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    K = min(PCA_DIM, Xc.shape[0], Xc.shape[1])
    atomic_savez(FACES_NPZ, X=Xa, ids=np.asarray(ids, dtype=np.int64),
                 mean=mean.astype(np.float32), comp=Vt[:K].astype(np.float32))


def detect_faces(img_path):
    """Находит лица на фото. Возвращает (картинка, [(прямоугольник, вектор)]) или (None, [])."""
    cv2 = _cv2()
    if cv2 is None:
        return None, []
    img = cv2.imread(img_path)
    if img is None:
        return None, []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # выравниваем освещение
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.15,
                                     minNeighbors=5, minSize=(40, 40))
    out = []
    for (x, y, w, h) in boxes:
        crop = cv2.resize(gray[y:y + h, x:x + w], (FACE_SIZE, FACE_SIZE))
        out.append(((int(x), int(y), int(w), int(h)),
                    crop.astype(np.float32).reshape(-1) / 255.0))
    return img, out


def _embed(v, mean, comp):
    return (v - mean) @ comp.T


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(a @ b / (na * nb))


def cmd_face_add(args):
    img, faces = detect_faces(args.photo)
    if img is None:
        print("Не смог открыть картинку:", args.photo)
        return
    if not faces:
        print("Лиц не нашёл. Нужно фото, где лицо смотрит в камеру и хорошо освещено.")
        return
    if len(faces) > 1 and not args.all:
        print(f"На фото {len(faces)} лиц. Какое добавить?")
        for i, (box, _) in enumerate(faces):
            print(f"  {i}: позиция x={box[0]} y={box[1]}, размер {box[2]}x{box[3]}")
        pick = input("Номер (Enter = 0, «all» = все): ").strip().lower()
        idxs = list(range(len(faces))) if pick == "all" else \
            [int(pick) if pick.isdigit() else 0]
    else:
        idxs = list(range(len(faces)))

    g = load_face_gallery()
    name = args.name.strip()
    p = g["people"].get(name)
    if p is None:
        p = {"id": g["next_id"], "count": 0}
        g["next_id"] += 1
        g["people"][name] = p
    X, ids = _load_gallery_npz()
    for i in idxs:
        X.append(faces[i][1])
        ids.append(p["id"])
        p["count"] += 1
    _save_gallery_npz(X, ids)
    save_face_gallery(g)
    print(f"🧑 Запомнил {len(idxs)} фото как «{name}». Теперь он знает {len(g['people'])} чел. в лицо.")
    if p["count"] < 3:
        print(f"   Совет: добавь ещё 2–4 фото «{name}» при другом свете/ракурсе — узнавать будет надёжнее.")
    print_trophies(check_milestones())


def cmd_face_find(args):
    img, faces = detect_faces(args.photo)
    if img is None:
        print("Не смог открыть картинку:", args.photo)
        return
    if not faces:
        print("Лиц на фото не нашёл.")
        return
    g = load_face_gallery()
    if not g["people"] or not os.path.exists(FACES_NPZ):
        print("Гарольд ещё никого не знает. Познакомь: python harold.py face-add ИМЯ фото.jpg")
        return
    with np.load(FACES_NPZ) as z:
        X, ids = list(z["X"]), list(z["ids"])
        mean, comp = z["mean"], z["comp"]
    id2name = {p["id"]: nm for nm, p in g["people"].items()}
    gal = [(id2name[int(ids[j])], _embed(X[j], mean, comp)) for j in range(len(X))]
    print(f"Лиц на фото: {len(faces)}")
    for i, (box, v) in enumerate(faces):
        e = _embed(v, mean, comp)
        best_name, best_sim = None, -1.0
        for nm, ge in gal:
            s = _cos(e, ge)
            if s > best_sim:
                best_name, best_sim = nm, s
        if best_sim >= args.thr:
            print(f"  лицо {i} (x={box[0]}, y={box[1]}): {best_name}  — уверенность {best_sim:.0%}")
        else:
            print(f"  лицо {i} (x={box[0]}, y={box[1]}): не знаю такого "
                  f"(ближайшее: {best_name}, {best_sim:.0%})")
    print_trophies(check_milestones())


def cmd_face_list(args):
    g = load_face_gallery()
    if not g["people"]:
        print("Пока никого не знает. Познакомь: python harold.py face-add ИМЯ фото.jpg")
        return
    print("———— 👥 Гарольд знает в лицо ————————")
    for nm, p in sorted(g["people"].items(), key=lambda kv: kv[1]["id"]):
        print(f"  {nm}  ({p['count']} фото)")


def cmd_face_forget(args):
    g = load_face_gallery()
    name = args.name.strip()
    if name not in g["people"]:
        print("Такого человека он и не знает. Список: python harold.py face-list")
        return
    pid = g["people"][name]["id"]
    X, ids = _load_gallery_npz()
    keep = [j for j in range(len(ids)) if int(ids[j]) != pid]
    if keep:
        _save_gallery_npz([X[j] for j in keep], [int(ids[j]) for j in keep])
    else:
        _save_gallery_npz([], [])
    del g["people"][name]
    save_face_gallery(g)
    print(f"Забыл «{name}» навсегда.")


# ============================ САЙТ: Гарольд в браузере ============================

WEB_INDEX = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Мистер Гарольд — чат</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --line:#30363d; --txt:#e6edf3; --dim:#8b949e; --acc:#58a6ff; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--txt); font:15px/1.5 "Segoe UI",system-ui,sans-serif; height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 18px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:18px; }
  #gen { color:var(--dim); font-size:13px; }
  #log { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:78%; padding:10px 14px; border-radius:14px; white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:#1f6feb33; border:1px solid #1f6feb66; }
  .bot  { align-self:flex-start; background:var(--card); border:1px solid var(--line); }
  footer { display:flex; gap:8px; padding:12px 18px; border-top:1px solid var(--line); }
  #q { flex:1; background:var(--card); border:1px solid var(--line); color:var(--txt); border-radius:10px; padding:10px 14px; font-size:15px; outline:none; }
  #q:focus { border-color:var(--acc); }
  button { background:var(--acc); color:#0d1117; border:0; border-radius:10px; padding:10px 18px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  #status { color:var(--dim); font-size:12px; padding:0 18px 8px; min-height:16px; }
</style>
</head>
<body>
<header><h1>🤖 Мистер Гарольд</h1><div id="gen">загрузка мозга…</div></header>
<main id="log"></main>
<div id="status"></div>
<footer>
  <input id="q" placeholder="Спроси Гарольда…" autocomplete="off">
  <button id="send">➤</button>
</footer>
<script>
"use strict";
const $ = s => document.querySelector(s);
const STOP = ["\nПользователь:", "\n\n"];
let M=null, W=null, IDX=null, vmap=null, rank=null, bpeCache=new Map();
let K=null, Vc=null, pos=0, ctxIds=[], busy=false;

const tick = () => new Promise(r => setTimeout(r, 0));
const scroll = () => { const l=$("#log"); l.scrollTop = l.scrollHeight; };

function addMsg(role, text){
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.textContent = text;
  $("#log").appendChild(d); scroll();
  return d;
}

async function boot(){
  try {
    M = await (await fetch("model.json")).json();
    const buf = await (await fetch(M.wfile)).arrayBuffer();
    W = new Float32Array(buf);
    IDX = {}; for (const w of M.weights) IDX[w.n] = w;
    vmap = {}; M.vocab.forEach((t,i)=> vmap[t]=i);
    if (M.tokenizer === "bpe") {
      rank = new Map();
      M.merges.forEach((m,i)=> rank.set(m[0]+"\u0000"+m[1], i));
    }
    $("#gen").textContent = "поколение " + M.generation + " · "
      + Math.round(M.n_params/1000) + "K параметров · "
      + (M.tokenizer === "bpe" ? "BPE" : "посимвольно")
      + " · контекст " + M.block_size
      + (M.block_size < 192 ? " (маловат: grow --ctx 256 и export-web заново)" : "");
    const about = M.memory && M.memory.about;
    addMsg("bot", "Привет! Я Мистер Гарольд. Мозг работает прямо в твоём браузере."
      + (about ? "\nЯ помню о тебе: " + about : ""));
  } catch(e) {
    $("#gen").textContent = "Ошибка загрузки: " + e.message + " — открой страницу через http:// (не file://)";
  }
}

function vec(name){ const e = IDX[name]; return W.subarray(e.o, e.o + e.s[0]*e.s[1]); }

function mat(x, name){
  const e = IDX[name], inD = e.s[1], outD = e.s[0];
  const w = W.subarray(e.o, e.o + outD*inD);
  const y = new Float32Array(outD);
  for (let j = 0; j < outD; j++){
    let b = j*inD, s = 0;
    for (let i = 0; i < inD; i++) s += x[i]*w[b+i];
    y[j] = s;
  }
  return y;
}

function rmsn(x, name){
  const D = M.d_model, g = vec(name);
  let s = 0; for (let i = 0; i < D; i++) s += x[i]*x[i];
  const inv = 1/Math.sqrt(s/D + 1e-5);
  const y = new Float32Array(D);
  for (let i = 0; i < D; i++) y[i] = g[i]*x[i]*inv;
  return y;
}

function step(id){
  const D = M.d_model, H = M.n_heads, Dh = D/H, L = M.n_layers, T = M.block_size;
  const scale = 1/Math.sqrt(Dh);
  const tokO = IDX["tok_emb"].o + id*D, posO = IDX["pos_emb"].o + pos*D;
  let h = new Float32Array(D);
  for (let i = 0; i < D; i++) h[i] = W[tokO+i] + W[posO+i];
  for (let l = 0; l < L; l++){
    const a1 = rmsn(h, l+".g1");
    const q = mat(a1, l+".Wq"), k = mat(a1, l+".Wk"), v = mat(a1, l+".Wv");
    K[l].set(k, pos*D); Vc[l].set(v, pos*D);
    const ya = new Float32Array(D);
    for (let hh = 0; hh < H; hh++){
      const off = hh*Dh, sc = new Float32Array(pos+1);
      let mx = -1e30;
      for (let t = 0; t <= pos; t++){
        let s = 0; const b = t*D + off;
        for (let i = 0; i < Dh; i++) s += q[off+i]*K[l][b+i];
        sc[t] = s*scale; if (sc[t] > mx) mx = sc[t];
      }
      let sum = 0;
      for (let t = 0; t <= pos; t++){ sc[t] = Math.exp(sc[t]-mx); sum += sc[t]; }
      for (let i = 0; i < Dh; i++){
        let s = 0;
        for (let t = 0; t <= pos; t++) s += sc[t]*Vc[l][t*D + off + i];
        ya[off+i] = s/sum;
      }
    }
    const at = mat(ya, l+".Wo");
    const h2 = new Float32Array(D);
    for (let i = 0; i < D; i++) h2[i] = h[i] + at[i];
    const a2 = rmsn(h2, l+".g2");
    const z = mat(a2, l+".W1");
    const b1 = vec(l+".b1");
    for (let i = 0; i < z.length; i++) z[i] = Math.max(0, z[i] + b1[i]);
    const m = mat(z, l+".W2");
    const b2 = vec(l+".b2");
    for (let i = 0; i < D; i++) h2[i] = h2[i] + m[i] + b2[i];
    h = h2;
  }
  const hf = rmsn(h, "gfin");
  const logits = mat(hf, "Wout");
  const bo = vec("bout");
  for (let i = 0; i < logits.length; i++) logits[i] += bo[i];
  pos++;
  return logits;
}

function sample(logits){
  const lg = new Float64Array(logits.length);
  for (let i = 0; i < lg.length; i++) lg[i] = logits[i];
  const recent = ctxIds.slice(-128);
  if (recent.length > 2){
    const seen = [...new Set(recent)];
    for (const t of seen) if (t < lg.length) lg[t] = lg[t] > 0 ? lg[t]/M.reppen : lg[t]*M.reppen;
  }
  for (let i = 0; i < lg.length; i++) lg[i] /= M.temp;
  let mx = -1e30; for (let i = 0; i < lg.length; i++) if (lg[i] > mx) mx = lg[i];
  const order = Array.from(lg.keys());
  order.sort((a,b) => lg[b] - lg[a]);
  const pr = new Float64Array(lg.length);
  let sum = 0;
  for (const i of order){ pr[i] = Math.exp(lg[i]-mx); sum += pr[i]; }
  let cum = 0, cut = order.length;
  for (let i = 0; i < order.length; i++){ cum += pr[order[i]]/sum; if (cum >= M.topp){ cut = i+1; break; } }
  const pool = order.slice(0, cut);
  let tot = 0; for (const t of pool) tot += pr[t];
  let r = Math.random()*tot;
  for (const t of pool){ r -= pr[t]; if (r <= 0) return t; }
  return pool[0];
}

function encode(s){
  if (M.tokenizer === "char"){
    const out = [];
    for (const c of s){ const i = vmap[c]; if (i !== undefined) out.push(i); }
    return out;
  }
  const out = [];
  const chunks = s.match(/[\p{L}\p{N}_]+|\s+|[^\p{L}\p{N}_\s]/gu) || [];
  for (const ch of chunks){
    let syms;
    if (bpeCache.has(ch)) syms = bpeCache.get(ch);
    else {
      syms = [...ch];
      while (syms.length > 1){
        let bi = -1, br = Infinity;
        for (let i = 0; i < syms.length-1; i++){
          const r = rank.get(syms[i]+"\u0000"+syms[i+1]);
          if (r !== undefined && r < br){ br = r; bi = i; }
        }
        if (bi < 0) break;
        syms.splice(bi, 2, syms[bi]+syms[bi+1]);
      }
      bpeCache.set(ch, syms);
    }
    for (const sym of syms){ const i = vmap[sym]; if (i !== undefined) out.push(i); }
  }
  return out;
}

function decodeIds(ids){ let s = ""; for (const i of ids) s += (M.vocab[i] ?? ""); return s; }

function earlyStop(t){
  let cut = null;
  for (const st of STOP){ const i = t.indexOf(st); if (i !== -1 && (cut === null || i < cut)) cut = i; }
  return cut;
}

function buildPrompt(msg){
  let head = "";
  if (M.memory && M.memory.about) head += "\n" + M.memory.about;
  for (const e of (M.memory && M.memory.episodes) || [])
    head += "\nПользователь: " + e.user + "\nГарольд: " + e.harold;
  let ids = encode(head + "\nПользователь: " + msg + "\nГарольд:");
  const max = M.block_size - 8;
  if (ids.length > max) ids = ids.slice(ids.length - max);
  return ids;
}

async function generate(msg){
  const T = M.block_size, D = M.d_model, L = M.n_layers;
  K = []; Vc = [];
  for (let l = 0; l < L; l++){ K.push(new Float32Array(T*D)); Vc.push(new Float32Array(T*D)); }
  pos = 0;
  const ids = buildPrompt(msg);
  ctxIds = ids.slice();
  const outEl = addMsg("bot", "…");
  let logits = null;
  for (const id of ids){ logits = step(id); await tick(); }
  const maxNew = Math.min(180, T - ctxIds.length - 1);
  const gen = [];
  for (let n = 0; n < maxNew; n++){
    const id = sample(logits);
    ctxIds.push(id); gen.push(id);
    const dec = decodeIds(gen);
    const cut = earlyStop(dec);
    if (cut !== null){ outEl.textContent = dec.slice(0, cut); break; }
    outEl.textContent = dec; scroll();
    if (pos >= T - 1) break;
    logits = step(id);
    await tick();
  }
  if (!outEl.textContent) outEl.textContent = "…";
}

async function send(){
  if (busy) return;
  const msg = $("#q").value.trim();
  if (!msg) return;
  $("#q").value = "";
  addMsg("user", msg);
  busy = true; $("#send").disabled = true; $("#q").disabled = true;
  $("#status").textContent = "Гарольд думает…";
  try { await generate(msg); }
  catch(e){ addMsg("bot", "⚠️ " + e.message); }
  $("#status").textContent = "";
  busy = false; $("#send").disabled = false; $("#q").disabled = false; $("#q").focus();
}

 $("#send").addEventListener("click", send);
 $("#q").addEventListener("keydown", e => { if (e.key === "Enter") send(); });
boot();
</script>
</body>
</html>
"""


def cmd_export_web(args):
    model = load_model()
    if model is None:
        print("Сначала: python harold.py init")
        return
    text = read_corpus()
    if text:
        ensure_vocab(model, text)
    out = args.out
    os.makedirs(out, exist_ok=True)
    entries = []
    blob = bytearray()

    def add(name, arr, transpose=False):
        a = np.asarray(arr, dtype=np.float32)
        if transpose:
            a = np.ascontiguousarray(a.T)   # JS-матвектору удобнее транспонированная
        entries.append({"n": name, "s": list(a.shape), "o": len(blob)})
        blob.extend(a.astype("<f4", copy=False).tobytes())

    add("tok_emb", model.params["tok_emb"])
    add("pos_emb", model.params["pos_emb"])
    add("gfin", model.params["gfin"])
    add("bout", model.params["bout"])
    add("Wout", model.params["Wout"], transpose=True)
    for l in range(model.cfg["n_layers"]):
        for nm in ("g1", "g2", "b1", "b2"):
            add(f"{l}.{nm}", model.params[f"{l}.{nm}"])
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            add(f"{l}.{nm}", model.params[f"{l}.{nm}"], transpose=True)
        add(f"{l}.W1", model.params[f"{l}.W1"], transpose=True)
        add(f"{l}.W2", model.params[f"{l}.W2"], transpose=True)

    with open(os.path.join(out, "harold.bin"), "wb") as f:
        f.write(bytes(blob))

    memory = None
    if args.memory:
        p = load_profile()
        eps = load_episodes()[-max(0, args.episodes):]
        memory = {"about": memory_static_text(p),
                  "episodes": [{"user": e["user"], "harold": e["harold"]} for e in eps]}

    header = {
        "d_model": model.cfg["d_model"], "n_heads": model.cfg["n_heads"],
        "n_layers": model.cfg["n_layers"], "block_size": model.cfg["block_size"],
        "tokenizer": model.tok.kind, "vocab": model.vocab,
        "merges": [list(m) for m in model.tok.merges],
        "weights": entries, "generation": model.generation,
        "n_params": n_params(model), "wfile": "harold.bin",
        "temp": 0.8, "topp": 0.92, "reppen": 1.12,
        "memory": memory,
    }
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8") as f:
        json.dump(header, f, ensure_ascii=False)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(WEB_INDEX)

    mb = len(blob) / 1e6
    print(f"🌐 Клон Гарольда в браузере собран в «{out}/»: index.html + model.json + harold.bin ({mb:.1f} МБ)")
    if args.memory:
        print("   ⚠️ В страницу вшита ЛИЧНАЯ ПАМЯТЬ — публиковать только в приватном доступе!")
        print("     Для публичного сайта (бесплатный Pages) собирай без --memory")
    print(f"   Локальный просмотр:  python -m http.server 8000 -d {out}")
    print("   Затем открой http://localhost:8000 — Гарольд отвечает прямо в браузере.")
    print_trophies(check_milestones())


# ============================ ОБЛАКО: GitHub Actions ============================

WORKFLOW = """name: Harold — облачная тренировка
on:
  workflow_dispatch:
    inputs:
      gens:
        description: 'Сколько поколений за запуск'
        default: '5'
        required: false
      steps:
        description: 'Шагов на поколение'
        default: '600'
        required: false
  schedule:
    - cron: '0 3 * * *'

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  train:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Зависимости
        run: pip install numpy pillow
      - name: Тренировка поколений
        run: |
          python harold.py evolve --gens "${{ github.event.inputs.gens || '5' }}" --steps "${{ github.event.inputs.steps || '600' }}" --read 1 --lr 1e-3 --auto-grow
      - name: Сборка сайта-клона
        run: python harold.py export-web --out site
      - name: Коммит весов
        run: |
          git config user.name "harold-bot"
          git config user.email "actions@users.noreply.github.com"
          G=$(python -c "import json;print(json.load(open('harold_data/harold.json'))['generation'])")
          git add harold_data site
          git commit -m "Гарольд: поколение $G [skip ci]" || echo "нечего коммитить"
          git push
      - name: Публикация на GitHub Pages
        uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - name: Deploy
        uses: actions/deploy-pages@v4
"""


def cmd_cloud_setup(args):
    wf_dir = os.path.join(".github", "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    wf = os.path.join(wf_dir, "harold.yml")
    with open(wf, "w", encoding="utf-8") as f:
        f.write(WORKFLOW)
    gi_lines = ["__pycache__/", "*.tmp", "harold_data/backups/", "harold_data/mnist/",
                "harold_data/faces/"]
    gi_path = ".gitignore"
    existing = set()
    if os.path.exists(gi_path):
        with open(gi_path, encoding="utf-8") as f:
            existing = {l.strip() for l in f}
    with open(gi_path, "a", encoding="utf-8") as f:
        for l in gi_lines:
            if l not in existing:
                f.write(l + "\n")
        f.write("# harold_data/memory/   ← раскомментируй, если репозиторий ПУБЛИЧНЫЙ\n")
    print("🌩 Workflow создан: .github/workflows/harold.yml (+ .gitignore дополнен, фотоальбом скрыт)")
    print("""
Дальше — 5 шагов:
  1) git init && git add harold.py .gitignore .github harold_data && git commit -m "Гарольд идёт в облако"
  2) Создай репозиторий на github.com (Public — для бесплатного Pages).
     Публичный? → раскомментируй "# harold_data/memory/" в .gitignore и перезапусти git add.
  3) git remote add origin https://github.com/ТВОЙ_НИК/ИМЯ.git
     git branch -M main
     git push -u origin main
  4) Settings → Pages → Source: «GitHub Actions»
  5) Actions → «Harold — облачная тренировка» → Run workflow (gens = 10)
""")


# ============================ панель и дневник ============================

def cmd_stats(args):
    if not os.path.exists(META_PATH):
        print("Гарольд ещё не создан: python harold.py init")
        return
    check_milestones()
    c = gather_context()
    xp = compute_xp(c)
    idx, title, cur_thr, next_thr = level_for(xp)
    intellect = compute_intellect(c)
    print("═" * 66)
    print("  🤖 МИСТЕР ГАРОЛЬД — панель развития".center(62))
    print("═" * 66)
    age = "?"
    if c["born"]:
        try:
            born = time.mktime(time.strptime(c["born"], "%Y-%m-%d %H:%M:%S"))
            days = max((time.time() - born) / 86400, 0)
            age = f"{days:.1f} дн." if days < 30 else f"{days / 30:.1f} мес."
        except Exception:
            pass
    print(f"  Уровень {idx} «{title}»    XP {fmt_int(xp)}")
    if next_thr:
        frac = (xp - cur_thr) / (next_thr - cur_thr)
        print(f"  до уровня {idx + 1}: {fmt_int(next_thr - xp)} XP  [{bar(frac, 30)}]")
    else:
        print(f"  Максимальный уровень! [{bar(1.0, 30)}]")
    print(f"  Поколение {c['gens']} · возраст {age} · интеллект {intellect}/100"
          f" [{bar(intellect / 100, 20)}] {intellect_label(intellect)}")
    print("—" * 66)
    print("  🧠 МОЗГ")
    arch = c["arch"]
    tok_str = f"BPE ({c['merges']} слияний)" if c["tok_kind"] == "bpe" else "посимвольный"
    print(f"   параметров     {fmt_int(c['params'])}   токенизатор: {tok_str}")
    if arch:
        print(f"   архитектура    d_model={arch.get('d_model')} · слоёв {arch.get('n_layers')}"
              f" · голов {arch.get('n_heads')} · контекст {arch.get('block_size')}")
    fill = min(c["corpus"] / max(c["params"] * 18, 1), 1.5)
    state = "отлично усваивает" if fill < 0.7 else ("плотно заполнен" if fill < 1 else "ПЕРЕПОЛНЕН!")
    print(f"   заполнение     [{bar(fill / 1.5, 30)}] {state}")
    if fill >= 0.9:
        print("   💡 python harold.py grow --mlp 2  или  evolve --auto-grow")
    lr_now = c["arch"].get("lr")
    if lr_now is not None and lr_now < 3.5e-4:
        print(f"   ⚠️ темп обучения задавлен (lr={lr_now:.1e}) — верни: train --lr 1e-3")
    print("  📚 ИСТОЧНИКИ ЗНАНИЙ")
    print(f"   Википедия {c['wiki']} · SCP {c['scp']} · веб-страницы {c['web']}"
          f" · книги {c['books']} · файлов {c['files']}")
    print(f"   прочитано      {fmt_int(c['corpus'])} символов")
    print(f"   лексикон       {c['vocab']} токенов · {fmt_int(c['words'])} слов · уроки {c['dialogs']}")
    if c["losses"]:
        print(f"   ошибка         {max(c['losses']):.2f} → {min(c['losses']):.2f}   {spark(c['losses'])}")
    if c["val_losses"]:
        print(f"   контрольная    {spark(c['val_losses'])}   сейчас {c['val_losses'][-1]:.3f}")
    print("  🚀 АНТИ-ПОТОЛОК")
    bv = c["best_val"]
    print(f"   рекорд понимания {bv if bv is None else f'{bv:.3f}'} · "
          f"авто-ростов {c['auto_grows']} · откатов порчи {c['rollbacks']}")
    k, v = GROWTH_LADDER[c["ladder_idx"] % len(GROWTH_LADDER)]
    nxt = {"ctx": f"контекст → {v}", "mlp": f"мышление ×{v}", "depth": f"+{v} слой"}[k]
    print(f"   следующая ступень роста: {nxt}")
    print("  👁 ГЛАЗА И ЛИЦА")
    vis_str = f"{c['vis'] * 100:.1f}%" if c["vis"] is not None else "цифры не обучены (vision-train)"
    print(f"   цифры          {vis_str}")
    print(f"   лица           знает {c.get('faces', 0)} чел. (face-add / face-find)")
    print("  💭 ПАМЯТЬ (никогда не забудет)")
    p = c["profile"]
    print(f"   о хозяине      {memory_static_text(p) or '—'}")
    print(f"   воспоминаний   {c['episodes']} разговоров")
    q = c["quiz"]
    print(f"   экзамен        {'вспоминает %.0f%% уроков' % (q * 100) if q is not None else 'не сдавал (quiz)'}")
    print("  ☕ БЕСЕДЫ И ТРУД")
    s = c["stats"]
    print(f"   сообщений {s.get('messages', 0)} · бесед {s.get('chats', 0)}"
          f" · шагов {fmt_int(s.get('total_steps', 0))}"
          f" · за пультом {fmt_dur(s.get('train_seconds', 0))}"
          f" · облачных запусков {s.get('cloud_runs', 0)}")
    unlocked = c["milestones"]
    titles = [t for mid, t, _ in MILESTONES if mid in unlocked]
    print(f"  🏆 ТРОФЕИ {len(titles)}/{len(MILESTONES)}")
    print("   " + (" · ".join(titles[-10:]) if titles else "—"))
    print("═" * 66)
    print("  Рецепты: crawl scp · feed папка · evolve --lr 1e-3 --auto-grow · face-add · export-web · cloud-setup")


def cmd_diary(args):
    diary = read_diary()
    if not diary:
        print("Дневник пуст. Запускайте harold evolve — записи появятся здесь.")
        return
    if args.chart:
        losses = [e["train_loss"] for e in diary if e.get("train_loss") is not None]
        vloss = [e["val_loss"] for e in diary if e.get("val_loss") is not None]
        corp = [e["corpus_chars"] for e in diary if e.get("corpus_chars")]
        par = [e.get("params") for e in diary if e.get("params")]
        ctx = [e.get("ctx") for e in diary if e.get("ctx")]
        print("———— 📈 Графики жизни Гарольда ————————")
        if losses:
            print(f"  ошибка     {losses[0]:.2f} → {losses[-1]:.2f}  {spark(losses, 56)}")
        if vloss:
            print(f"  контроль   {vloss[0]:.2f} → {vloss[-1]:.2f}  {spark(vloss, 56)}")
        if corp:
            print(f"  корпус     {fmt_int(corp[0])} → {fmt_int(corp[-1])}  {spark(corp, 56)}")
        if par:
            print(f"  мозг       {fmt_int(par[0])} → {fmt_int(par[-1])}  {spark(par, 56)}")
        if ctx:
            print(f"  контекст   {ctx[0]} → {ctx[-1]} токенов")
        print()
    for e in diary[-args.n:]:
        if e.get("event"):
            print(f"[{e.get('time', '')}] ⚡ {e['event']}")
            continue
        vl = f"{e['val_loss']:.3f}" if e.get("val_loss") is not None else "—"
        print(f"поколение {e.get('generation', 0):>3} | ошибка {e.get('train_loss', 0):.3f}"
              f" | контроль {vl} | шагов {e.get('steps', '?')}"
              f" | контекст {e.get('ctx', '?')} | корпус {e.get('corpus_chars', 0):>9,}"
              f" | «{e.get('sample', '')[:44]}»")


def cmd_reset(args):
    if os.path.exists(DATA_DIR):
        print("⚠️  Будет удалено ВСЁ: мозг, знания, ВЕЧНАЯ ПАМЯТЬ, фотоальбом, трофеи, бэкапы.")
        if input(f"Удалить {DATA_DIR} полностью? (y/n): ").strip().lower() == "y":
            shutil.rmtree(DATA_DIR)
            print("Гарольд удалён. Создайте нового: python harold.py init")
    else:
        print("Нечего удалять.")


# ============================ CLI ============================

def main():
    ap = argparse.ArgumentParser(description="Мистер Гарольд — развивающаяся нейросеть")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="создать Гарольда с нуля")
    sp.add_argument("--d", type=int, default=192)
    sp.add_argument("--layers", type=int, default=3)
    sp.add_argument("--heads", type=int, default=4)
    sp.add_argument("--ctx", type=int, default=192)
    sp.add_argument("--bpe", action="store_true")
    sp.add_argument("--merges", type=int, default=2000)
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser("rebirth", help="ПЕРЕРОЖДЕНИЕ: новый больший мозг, душа остаётся")
    sp.add_argument("--d", type=int, default=192)
    sp.add_argument("--layers", type=int, default=3)
    sp.add_argument("--heads", type=int, default=4)
    sp.add_argument("--ctx", type=int, default=256)
    sp.add_argument("--bpe", action="store_true")
    sp.add_argument("--merges", type=int, default=2000)
    sp.set_defaults(fn=cmd_rebirth)

    sp = sub.add_parser("crawl", help="добыча знаний: SCP и списки ссылок")
    csub = sp.add_subparsers(dest="source", required=True)
    s1 = csub.add_parser("scp", help="статьи Фонда SCP")
    s1.add_argument("--base", default="https://scp-ru.wikidot.com")
    s1.add_argument("--start", type=int, default=1)
    s1.add_argument("--end", type=int, default=300)
    s1.add_argument("--suffix", default="")
    s1.add_argument("--delay", type=float, default=2.5)
    s1.add_argument("--min_chars", type=int, default=800)
    s1.set_defaults(fn=cmd_crawl)
    s2 = csub.add_parser("urls", help="файл со ссылками, по одной в строке")
    s2.add_argument("file")
    s2.add_argument("--delay", type=float, default=1.5)
    s2.set_defaults(fn=cmd_crawl)

    sp = sub.add_parser("read", help="случайные статьи Википедии")
    sp.add_argument("--n", type=int, default=3)
    sp.add_argument("--lang", default="ru")
    sp.set_defaults(fn=cmd_read)

    sp = sub.add_parser("learn", help="прочитать одну веб-страницу")
    sp.add_argument("--url", required=True)
    sp.set_defaults(fn=cmd_learn)

    sp = sub.add_parser("ingest", help="добавить книгу/файл (повторное кормление безопасно)")
    sp.add_argument("file")
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("feed", help="скормить папку .txt или файл")
    sp.add_argument("path")
    sp.set_defaults(fn=cmd_feed)

    sp = sub.add_parser("teach", help="научить вопросу-ответу")
    sp.set_defaults(fn=cmd_teach)

    sp = sub.add_parser("train", help="одна тренировка")
    sp.add_argument("--steps", type=int, default=500)
    sp.add_argument("--batch", type=int, default=8)
    sp.add_argument("--accum", type=int, default=1)
    sp.add_argument("--lr", type=float, default=None)
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("evolve", help="эволюция поколениями (главная команда)")
    sp.add_argument("--gens", type=int, default=5)
    sp.add_argument("--steps", type=int, default=600)
    sp.add_argument("--batch", type=int, default=8)
    sp.add_argument("--accum", type=int, default=1)
    sp.add_argument("--read", type=int, default=1)
    sp.add_argument("--lang", default="ru")
    sp.add_argument("--goal", type=float, default=None)
    sp.add_argument("--lr", type=float, default=None,
                    help="задать темп обучения (сохранится в конфиге)")
    sp.add_argument("--auto-grow", dest="auto_grow", action="store_true")
    sp.add_argument("--patience", type=int, default=3)
    sp.set_defaults(fn=cmd_evolve)

    sp = sub.add_parser("grow", help="ручной нейрогенез")
    sp.add_argument("--depth", type=int, default=0)
    sp.add_argument("--mlp", type=int, default=0)
    sp.add_argument("--ctx", type=int, default=0)
    sp.set_defaults(fn=cmd_grow)

    sp = sub.add_parser("chat", help="поговорить с Гарольдом")
    sp.add_argument("--temp", type=float, default=0.8)
    sp.add_argument("--topk", type=int, default=0)
    sp.add_argument("--topp", type=float, default=0.92)
    sp.add_argument("--reppen", type=float, default=1.12)
    sp.add_argument("--max_new", type=int, default=200)
    sp.set_defaults(fn=cmd_chat)

    sp = sub.add_parser("say", help="продолжить фразу")
    sp.add_argument("prompt", nargs="?", default="Гарольд")
    sp.add_argument("--temp", type=float, default=0.8)
    sp.add_argument("--topk", type=int, default=0)
    sp.add_argument("--topp", type=float, default=0.92)
    sp.add_argument("--reppen", type=float, default=1.12)
    sp.add_argument("--max_new", type=int, default=150)
    sp.set_defaults(fn=cmd_say)

    sp = sub.add_parser("think", help="поток сознания")
    sp.add_argument("prompt", nargs="?", default="")
    sp.add_argument("--max_new", type=int, default=200)
    sp.set_defaults(fn=cmd_think)

    sp = sub.add_parser("remember", help="записать факт в вечную память")
    sp.add_argument("text", nargs="+")
    sp.set_defaults(fn=cmd_remember)

    sp = sub.add_parser("memory", help="что Гарольд помнит о тебе")
    sp.add_argument("--show", type=int, default=5)
    sp.set_defaults(fn=cmd_memory)

    sp = sub.add_parser("forget", help="забыть факты/воспоминания")
    sp.add_argument("--facts", action="store_true")
    sp.add_argument("--episodes", action="store_true")
    sp.add_argument("--name", action="store_true")
    sp.add_argument("--all", action="store_true")
    sp.set_defaults(fn=cmd_forget)

    sp = sub.add_parser("quiz", help="ЭКЗАМЕН: проверка памяти уроков")
    sp.add_argument("--n", type=int, default=40)
    sp.set_defaults(fn=cmd_quiz)

    sp = sub.add_parser("vision-train", help="научить видеть цифры (MNIST)")
    sp.add_argument("--epochs", type=int, default=2)
    sp.set_defaults(fn=cmd_vision_train)

    sp = sub.add_parser("see", help="распознать цифру на картинке")
    sp.add_argument("image")
    sp.set_defaults(fn=cmd_see)

    sp = sub.add_parser("face-add", help="познакомить Гарольда с человеком по фото")
    sp.add_argument("name")
    sp.add_argument("photo")
    sp.add_argument("--all", action="store_true", help="взять все лица с фото")
    sp.set_defaults(fn=cmd_face_add)

    sp = sub.add_parser("face-find", help="кто на фото?")
    sp.add_argument("photo")
    sp.add_argument("--thr", type=float, default=0.45, help="порог узнавания (0..1)")
    sp.set_defaults(fn=cmd_face_find)

    sp = sub.add_parser("face-list", help="кого Гарольд знает в лицо")
    sp.set_defaults(fn=cmd_face_list)

    sp = sub.add_parser("face-forget", help="забыть человека")
    sp.add_argument("name")
    sp.set_defaults(fn=cmd_face_forget)

    sp = sub.add_parser("export-web", help="собрать сайт-чат: Гарольд в браузере")
    sp.add_argument("--out", default="site")
    sp.add_argument("--memory", action="store_true",
                    help="вшить память о хозяине (только для приватной публикации!)")
    sp.add_argument("--episodes", type=int, default=3)
    sp.set_defaults(fn=cmd_export_web)

    sp = sub.add_parser("cloud-setup", help="создать GitHub Actions: облако + Pages")
    sp.set_defaults(fn=cmd_cloud_setup)

    sp = sub.add_parser("stats", help="большая панель развития")
    sp.set_defaults(fn=cmd_stats)

    sp = sub.add_parser("diary", help="дневник развития")
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--chart", action="store_true")
    sp.set_defaults(fn=cmd_diary)

    sp = sub.add_parser("reset", help="стереть всё")
    sp.set_defaults(fn=cmd_reset)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()



#     git rm -r --cached книги -q
# git rm -r --cached harold_data/backups -q
# git rm -r --cached harold_data/mnist -q
# git rm -r --cached harold_data/faces -q



# # 1. yml на месте? Должен напечатать строку с "if: always()":
# findstr "always" .github\workflows\harold.yml

# # 2. Код жив?
# python harold.py -h