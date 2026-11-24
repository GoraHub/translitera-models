# Checks converted tc-big models with the same steps as the app (mirrors lib/translation/marian.dart).
import os, sys, json, time
src = open(os.path.join(os.path.dirname(__file__), "opus_check.py")).read()
exec(src.split("\nfor p in PAIRS")[0])  # Marian class and tokenizer loader, without the old run
cases = {
 "fi-zle": [(">>ukr<< ", s) for s in ["Hei, olen terveydenhoitaja Anna.", "Miten olette voineet raskauden aikana?",
     "Onko teillä ollut päänsärkyä tai turvotusta jaloissa?", "Mittaan nyt verenpaineen.",
     "Ottakaa yhteyttä synnytyssairaalaan, jos supistukset alkavat.", "Rokotus annetaan olkavarteen.",
     "Annamme tänään rokotteen. Pistoskohta voi olla punainen pari päivää."]] +
     [(">>rus<< ", s) for s in ["Mittaan nyt verenpaineen.", "Onko teillä kysyttävää?"]],
 "zle-fi": [("", s) for s in ["Мене звати Оксана.", "мене заґіати Оксана", "Чи потрібно мені здати аналіз крові?",
     "Мій чоловік не може прийти на прийом.", "Дякую, все зрозуміло.", "Меня зовут Оксана.",
     "Можно ли кормить грудью, если я принимаю антибиотики?"]],
}
base = sys.argv[1]
for pair, items in cases.items():
    d = f"{base}/Helsinki-NLP/opus-mt-tc-big-{pair}"
    print("\n==", pair, {f: round(os.path.getsize(f"{d}/onnx/{f}") / 1e6) for f in sorted(os.listdir(f"{d}/onnx")) if "quantized" in f})
    m = Marian(d)
    for pre, s in items:
        t = time.time(); g = m.translate(pre + s, beams=1); tg = int((time.time() - t) * 1000)
        t = time.time(); b = m.translate(pre + s, beams=4); tb = int((time.time() - t) * 1000)
        print(f"{pre}{s}\n   greedy {tg} ms: {g}\n   beam4  {tb} ms: {b}", flush=True)
