"""Pemuat prediktor untuk demo web.

KERANGKA SAJA: model IndoBERTweet MTL belum dilatih. Saat ini memakai
prediktor dummy (simulasi) supaya UI bisa diuji.

TODO (setelah eksperimen selesai):
- Ganti USE_DUMMY = False.
- Taruh checkpoint terbaik di ./models/ (atau set env MODEL_DIR):
  M1 MTL terbaik; jika M1 kalah, pakai model sentimen terbaik.
- Apakah parent_text dipakai mengikuti hasil eksperimen A2.
- Arsitektur: shared encoder IndoBERTweet + head sentimen (3) + head sarkasme (2), inferensi CPU.
"""
import hashlib
import math
import os
from pathlib import Path

USE_DUMMY = True
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models"))

_SENTIMENT_LABELS = ("positif", "negatif", "netral")
_SARCASM_LABELS = ("sarkasme", "non-sarkasme")

# Kata kunci kasar khusus simulasi; BUKAN bagian metode penelitian.
_POS = ("bagus", "berhasil", "senang", "mantap", "bermanfaat", "lanjutkan", "terima kasih", "bergizi")
_NEG = ("basi", "keracunan", "korupsi", "sakit", "buruk", "gagal", "mubazir", "kecewa", "khawatir", "dihentikan")
_SARC_CUE = ("bagus sekali", "hebat", "mantap sekali", "keren banget", "luar biasa", "ya kan", "wkwk")


def _softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = math.fsum(exps)
    return [e / total for e in exps]


class _DummyPredictor:
    """Simulasi deterministik: heuristik kata kunci + sedikit variasi hash."""

    def __call__(self, text, parent_text=None):
        return self.predict(text, parent_text)

    def predict(self, text, parent_text=None):
        t = ("" if text is None else str(text)).lower()
        p = ("" if parent_text is None else str(parent_text)).lower()
        digest = hashlib.sha256(f"{t}\0{p}".encode("utf-8")).digest()
        jitter = [(digest[i] / 255.0 - 0.5) * 0.6 for i in range(5)]

        pos = sum(k in t for k in _POS)
        neg = sum(k in t for k in _NEG) + 0.5 * sum(k in p for k in _NEG)
        cue = sum(k in t for k in _SARC_CUE)
        sarcastic_hint = cue > 0 and neg > 0  # pujian + konteks buruk

        if sarcastic_hint:
            pos_s, neg_s = 0.2, 2.0 + neg
        else:
            pos_s, neg_s = 1.5 * pos, 1.5 * neg
        neu_s = 1.0 if (pos == 0 and neg == 0) else 0.3
        s_probs = _softmax([pos_s + jitter[0], neg_s + jitter[1], neu_s + jitter[2]])

        sarc_s = 2.0 if sarcastic_hint else (0.6 if cue else -1.0)
        c_probs = _softmax([sarc_s + jitter[3], 0.0 + jitter[4]])

        sentiment_probs = dict(zip(_SENTIMENT_LABELS, s_probs))
        sarcasm_probs = dict(zip(_SARCASM_LABELS, c_probs))
        return {
            "sentiment_label": max(sentiment_probs, key=sentiment_probs.get),
            "sentiment_probs": sentiment_probs,
            "sarcasm_label": max(sarcasm_probs, key=sarcasm_probs.get),
            "sarcasm_probs": sarcasm_probs,
        }


def load_predictor():
    if not USE_DUMMY:
        # TODO: load tokenizer + checkpoint MTL dari MODEL_DIR di sini.
        raise RuntimeError(
            "Checkpoint model belum terintegrasi. Mode model nyata tidak tersedia."
        )
    return _DummyPredictor()


def predict(text, parent_text=None):
    return load_predictor().predict(text, parent_text)
