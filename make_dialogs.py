#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Фабрика диалогов: превращает статьи Википедии из корпуса в пары вопрос-ответ.
Создаёт harold_data/corpus/98_dialogs_wiki.txt
Запуск: python make_dialogs.py"""
import os, re

CORPUS = os.path.join("harold_data", "corpus")
OUT = os.path.join(CORPUS, "98_dialogs_wiki.txt")

QUESTIONS = [
    "расскажи про {t}",
    "что такое {t}?",
    "кто такой {t}?",
    "расскажи, что ты знаешь про {t}",
    "что ты читал про {t}?",
    "расскажи что-нибудь про {t}",
    "знаешь что-нибудь про {t}?",
    "расскажи про {t}, папа велел узнать",
]


def sentences(t):
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if len(p.strip()) > 25]


total = 0
out_lines = []
for fn in sorted(os.listdir(CORPUS)):
    if not fn.endswith(".txt"):
        continue
    with open(os.path.join(CORPUS, fn), encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for block in re.split(r"(?=Статья Википедии: )", text):
        m = re.match(r"Статья Википедии: (.+?)\.\s*", block)
        if not m:
            continue
        title = m.group(1).strip()
        body = block[m.end():]
        sents = sentences(body)
        if not sents:
            continue
        answer = " ".join(sents[:3])[:400]
        q = QUESTIONS[total % len(QUESTIONS)].format(t=title)
        out_lines.append(f"\nПользователь: {q}\nГарольд: {answer}")
        total += 1

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")
print(f"Создано диалогов: {total} → {OUT}")