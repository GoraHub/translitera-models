# The transformers.js converter writes the Marian word list in SentencePiece (alphabetical) order,
# but the model numbers words by its own vocab.json. Rebuild tokenizer.json in the model's order,
# with SentencePiece scores from the source-language model (words it can't produce get a very low score).
import json, sys
from transformers import MarianTokenizer
name, path = sys.argv[1], sys.argv[2]
ht = MarianTokenizer.from_pretrained(name)
sp = ht.spm_source
score = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())}
by_id = sorted(ht.encoder.items(), key=lambda kv: kv[1])
assert [i for _, i in by_id] == list(range(len(by_id))), "vocab ids are not contiguous"
special = {"</s>", "<unk>", "<pad>"}
tj = json.load(open(path))
tj["model"]["vocab"] = [[p, 0.0 if p in special else score.get(p, -1000.0)] for p, _ in by_id]
tj["model"]["unk_id"] = ht.encoder["<unk>"]
for a in tj.get("added_tokens", []):
    a["id"] = ht.encoder[a["content"]]
# target-language tokens such as >>ukr<< are whole tokens, never split
import re
have = {a["content"] for a in tj.get("added_tokens", [])}
for p, i in by_id:
    if re.fullmatch(r">>\w+<<", p) and p not in have:
        tj.setdefault("added_tokens", []).append({"id": i, "content": p, "single_word": False, "lstrip": False,
                                                 "rstrip": False, "normalized": False, "special": True})
pp = tj.get("post_processor") or {}
for k, v in (pp.get("special_tokens") or {}).items():
    v["ids"] = [ht.encoder[k]]
json.dump(tj, open(path, "w"), ensure_ascii=False)
# check: same ids as the original tokenizer
from tokenizers import Tokenizer
n = tj.get("normalizer")
if n and n.get("type") == "Precompiled" and not n.get("precompiled_charsmap"):
    tj["normalizer"] = None
tok = Tokenizer.from_str(json.dumps(tj))
tests = ["Мене звати Оксана.", "Меня зовут Оксана, мне 32 года.", "Miten olette voineet raskauden aikana?",
         "Rokotus annetaan olkavarteen 12.3.", ">>ukr<< Mittaan nyt verenpaineen.", ">>rus<< Onko teillä kysyttävää?", "Дитина погано спить уночі, температура 38,5."]
bad = 0
tests = [t for t in tests if not t.startswith(">>") or t.split()[0] in ht.encoder]
for t in tests:
    a, b = ht(t)["input_ids"], tok.encode(t).ids
    print("ok " if a == b else "DIFF", t, a if a == b else (a, b))
    bad += a != b
sys.exit(1 if bad else 0)
