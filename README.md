# Translitera models

Translation models that the Translitera app downloads to the phone. Translation then runs fully on the phone.
This repository holds only open models and the scripts that build them. It contains no app code and no user data.

| Files | Model | Direction |
|---|---|---|
| `tc-big-fi-zle__*` | [Helsinki-NLP/opus-mt-tc-big-fi-zle](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-fi-zle) | Finnish → Ukrainian / Russian |
| `tc-big-zle-fi__*` | [Helsinki-NLP/opus-mt-tc-big-zle-fi](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-zle-fi) | Ukrainian / Russian → Finnish |

## Licence and attribution

The models are by the Language Technology Research Group at the University of Helsinki
(Tiedemann et al., OPUS-MT and the Tatoeba Translation Challenge), and are licensed under
[CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Changes: converted to ONNX with the transformers.js 3.7.0 scripts, with weights quantized to 8-bit integers.
The tokenizer word list was reordered to match the model's own numbering.
The workflow in `.github/workflows/build-models.yml` produces every released file.

- Tiedemann, J. and Thottingal, S. (2020). OPUS-MT – Building open translation services for the World. EAMT 2020.
- Tiedemann, J., Aulamo, M., Bakshandaeva, D., Boggia, M., Grönroos, S.-A., Nieminen, T., Raganato, A., Scherrer, Y., Vázquez, R. and Virpioja, S. (2023). Democratizing neural machine translation with OPUS-MT. Language Resources and Evaluation.
