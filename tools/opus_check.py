"""Cloud check for the on-device OPUS-MT pipeline.
Downloads the models the app uses, translates neuvola test sentences with the SAME steps as
lib/translation/marian.dart (encoder, decoder_model, then decoder_with_past_model, greedy, no <pad>),
and writes tokenizer reference vectors for the Dart test."""
import json, os, sys, urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

OUT = os.environ.get("OPUS_DIR", "/tmp/opus")
FILES = ["config.json", "tokenizer.json", "onnx/encoder_model_quantized.onnx",
         "onnx/decoder_model_quantized.onnx", "onnx/decoder_with_past_model_quantized.onnx"]
PAIRS = ["fi-en", "en-fi", "uk-en", "en-uk", "ru-en", "en-ru"]
SENT = {
    "fi": ["Miten vauva on nukkunut viime aikoina?", "Onko vauvalla ollut kuumetta tai ripulia?",
           "Annamme tänään rokotteen. Pistoskohta voi olla punainen pari päivää.",
           "Seuraava käynti on kolmen kuukauden päästä."],
    "uk": ["Дитина погано спить уночі.", "У неї була температура тридцять вісім.", "Скільки разів на день її годувати?"],
    "ru": ["Ребёнок плохо спит ночью.", "У неё была температура тридцать восемь.", "Сколько раз в день её кормить?"],
    "en": ["How has the baby slept lately?", "We will give the vaccine today."],
}

def fetch(pair):
    d = os.path.join(OUT, pair)
    for f in FILES:
        p = os.path.join(d, f)
        if os.path.exists(p): continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        url = f"https://huggingface.co/Xenova/opus-mt-{pair}/resolve/main/{f}"
        urllib.request.urlretrieve(url, p)
        print("downloaded", pair, f, os.path.getsize(p) // 1024, "KB")
    return d

def _strip_empty_precompiled(n):
    # Some exports carry a Precompiled normalizer with an empty (null) table, which the Rust library
    # rejects. It does nothing, so drop it (the Dart tokenizer ignores it too).
    if not n: return n
    if n.get("type") == "Precompiled" and not n.get("precompiled_charsmap"): return None
    if n.get("type") == "Sequence":
        ns = [x for x in (_strip_empty_precompiled(m) for m in n["normalizers"]) if x]
        return {"type": "Sequence", "normalizers": ns} if ns else None
    return n

def load_tokenizer(path):
    j = json.load(open(path))
    print("  tokenizer:", "normalizer", json.dumps(j.get("normalizer"))[:160], "| pre", json.dumps(j.get("pre_tokenizer"))[:160],
          "| decoder", json.dumps(j.get("decoder"))[:120], "| post", json.dumps(j.get("post_processor"))[:160])
    j["normalizer"] = _strip_empty_precompiled(j.get("normalizer"))
    return Tokenizer.from_str(json.dumps(j))

class Marian:
    def __init__(self, d):
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.tok = load_tokenizer(os.path.join(d, "tokenizer.json"))
        so = ort.SessionOptions()
        self.enc = ort.InferenceSession(os.path.join(d, "onnx/encoder_model_quantized.onnx"), so)
        self.dec = ort.InferenceSession(os.path.join(d, "onnx/decoder_model_quantized.onnx"), so)
        self.past = ort.InferenceSession(os.path.join(d, "onnx/decoder_with_past_model_quantized.onnx"), so)
        self.pad = cfg.get("pad_token_id", 58100)
        self.start = cfg.get("decoder_start_token_id", self.pad)
        self.eos = cfg.get("eos_token_id", 0)
        print("  inputs dec:", [i.name for i in self.dec.get_inputs()][:4], "... past:", [i.name for i in self.past.get_inputs()][:4])

    def translate(self, text, beams=1):
        """Same algorithm as MarianModel.translate in lib/translation/marian.dart."""
        ids = self.tok.encode(text).ids
        x = np.array([ids], dtype=np.int64); m = np.ones_like(x)
        hidden = self.enc.run(None, {"input_ids": x, "attention_mask": m})[0]
        hyps = [([self.start], 0.0, {})]; done = []
        for step in range(int(len(ids) * 2.5) + 10):
            if not hyps: break
            sess = self.dec if step == 0 else self.past
            names = [o.name for o in sess.get_outputs()]
            cands = []
            for toks, score, cache in hyps:
                feed = {}
                for i in sess.get_inputs():
                    n = i.name
                    if n == "input_ids": feed[n] = np.array([[toks[-1]]], dtype=np.int64)
                    elif n == "encoder_attention_mask": feed[n] = m
                    elif n == "encoder_hidden_states": feed[n] = hidden
                    elif n.startswith("past_key_values."): feed[n] = cache["present." + n[len("past_key_values."):]]
                    else: raise RuntimeError("unexpected input " + n)
                outs = sess.run(None, feed)
                c2 = dict(cache)
                for n, v in zip(names[1:], outs[1:]):
                    if n.startswith("present."): c2[n] = v
                row = outs[0][0, -1].astype(np.float64); row[self.pad] = -np.inf
                lp = row - np.logaddexp.reduce(row[np.isfinite(row)])
                k = 1 if beams == 1 else beams * 2
                for t in np.argsort(-lp)[:k]:
                    cands.append((toks + [int(t)], score + lp[t], c2))
            cands.sort(key=lambda c: -c[1])
            nxt = []
            for c in cands:
                if c[0][-1] == self.eos: done.append(c)
                elif len(nxt) < beams and not (len(c[0]) >= 5 and len(set(c[0][-4:])) == 1): nxt.append(c)
                if len(nxt) == beams: break
            if not nxt and not done: done.extend(cands[:1])
            hyps = nxt
            if len(done) >= beams: break
        pool = done or hyps
        pool.sort(key=lambda c: -(c[1] / len(c[0])))
        return self.tok.decode([t for t in pool[0][0][1:] if t != self.eos], skip_special_tokens=True)

models = {}
vectors = {}
for p in PAIRS:
    d = fetch(p)
    print("loading", p)
    models[p] = Marian(d)
    src = p.split("-")[0]
    vectors[p] = [{"text": s, "ids": models[p].tok.encode(s).ids, "decoded": models[p].tok.decode(models[p].tok.encode(s).ids, skip_special_tokens=True)} for s in SENT[src]]
json.dump(vectors, open(os.path.join(OUT, "vectors.json"), "w"), ensure_ascii=False, indent=1)

print("\n=== Translations (via English) ===")
for other in ["uk", "ru"]:
    for s in SENT["fi"]:
        en = models["fi-en"].translate(s); tgt = models[f"en-{other}"].translate(en, beams=4 if other == "uk" else 1)
        print(f"fi→{other}: {s}\n   en: {en}\n   {other}: {tgt}")
    for s in SENT[other]:
        en = models[f"{other}-en"].translate(s); fi = models["en-fi"].translate(en)
        print(f"{other}→fi: {s}\n   en: {en}\n   fi: {fi}")
