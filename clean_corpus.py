#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Чистка корпуса: выбрасывает статьи с чужими алфавитами (CJK, армянский,
тайский, иврит...). Оригиналы сохраняются в harold_data/backups.
Запуск: python clean_corpus.py"""
import os, re, shutil

CORPUS = os.path.join("harold_data", "corpus")
BACKUP = os.path.join("harold_data", "backups")
MARK = "Статья Википедии:"


def bad_ratio(t):
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 1.0
    ok = sum(1 for c in letters if ("a" <= c <= "z") or ("A" <= c <= "Z")
             or ("а" <= c <= "я") or c in "Ёё")
    return 1 - ok / len(letters)


total_kept = total_dropped = 0
for fn in sorted(os.listdir(CORPUS)):
    if not fn.endswith(".txt"):
        continue
    path = os.path.join(CORPUS, fn)
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if MARK not in text:
        continue
    parts = [p for p in re.split(r"(?=" + MARK + ")", text) if p.strip()]
    kept, dropped = [], 0
    for a in parts:
        if len(a) < 250 or bad_ratio(a) > 0.03:
            dropped += 1
        else:
            kept.append(a)
    if dropped == 0:
        print(f"{fn}: чисто ({len(kept)} статей)")
        continue
    os.makedirs(BACKUP, exist_ok=True)
    shutil.copy2(path, os.path.join(BACKUP, f"dirty_{fn}"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(kept) + "\n")
    total_kept += len(kept)
    total_dropped += dropped
    print(f"{fn}: оставлено {len(kept)}, выброшено {dropped} грязных статей")

print(f"\nИтого удалено {total_dropped} статей с чужими алфавитами (копии в backups).")