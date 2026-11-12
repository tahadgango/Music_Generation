# Music Generation with an LSTM

This project trains a character-level LSTM to continue music written in ABC notation. The main workflow is in a Jupyter notebook: it loads Irish traditional tunes, encodes their text as characters, trains a PyTorch next-character model, and samples new ABC text from a trained checkpoint.

## Project structure

```text
.
├── PT_Part2_Music_Generation.ipynb  # Data preparation, training, evaluation, and generation
├── src/
│   ├── data/irish.abc               # ABC notation training corpus
│   ├── bin/abc2wav                  # ABC-to-WAV helper script
│   └── music/
│       ├── generation.py            # Dataset loading, ABC parsing, and audio playback helpers
│       └── util.py                  # Plotting and training utilities
├── all_params.json                  # Example hyperparameter sweep configuration
├── params_list.json                 # Configurations consumed by the notebook
├── models/                          # Saved checkpoints, parameters, and loss plots
└── output/                          # Generated WAV files
```

## Requirements and setup

- Python 3.10 or newer is recommended.
- Install the Python dependencies listed in `requirements.txt`.
- To convert ABC files to audio, install the system commands `abc2midi` and `timidity` and make sure both are available on your `PATH`. They are external programs, not Python packages.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch the notebook from the project root:

```bash
jupyter notebook PT_Part2_Music_Generation.ipynb
```

## Model architecture

The model follows the LSTM sequence architecture illustrated below:

![Unrolled LSTM architecture](https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab1/img/lstm_unrolled-01-01.png)

Each input character is mapped by an embedding layer to a 256-dimensional vector. The unrolled LSTM processes the sequence one time step at a time, carrying hidden and cell state forward so later predictions can use context from earlier characters. At every time step, a linear output layer maps the LSTM state to a score (logit) for each character in the vocabulary. Training compares these logits with the next-character targets using cross-entropy loss. During generation, the model samples the next character from the output probabilities and feeds it back as the next input.

## Workflow

1. `generation.load_training_data()` reads tunes from `src/data/irish.abc`; the notebook builds a character vocabulary from the corpus.
2. The notebook converts the corpus to integer IDs and forms input sequences and targets shifted forward by one character.
3. A PyTorch model (embedding → LSTM → linear projection) is trained and evaluated using next-character cross-entropy loss.
4. The notebook saves the best validation checkpoint, its parameters, and a loss plot in a run-specific directory under `models/`.
5. It loads a selected checkpoint, samples a continuation from an ABC header, and can convert syntactically valid generated tunes to WAV files in `output/`.

### Device selection

The notebook automatically selects the available compute device in this order: CUDA GPU, Apple Metal (MPS), then CPU. It uses the first available option, so no device setting is needed for the standard workflow.

## Hyperparameter experiments and results

The checked-in experiments use batch size **128**, sequence length **100**, **3,000** training iterations, and embedding dimension **256**. The table reports each run's best saved validation cross-entropy loss (lower is better). A dash means that combination is not present in the recorded results.

| LSTM hidden size | LR 0.005 | LR 0.001 | LR 0.0005 | LR 0.0001 |
|---:|---:|---:|---:|---:|
| 1024 | 1.420445 | 1.383801 | — | — |
| 512  | 1.415073 | 1.378657 | — | — |
| 256  | 1.413317 | 1.373872 | **1.368786** | **1.371095** |
| 128  | 1.450794 | 1.406509 | 1.402358 | 1.479724* |
| 64   | 1.482318 | 1.470446 | 1.482378 | 4.418365* |
| 32   | 1.577476 | 1.551635 | 1.590175 | 1.801560* |

\* These 0.0001 runs used **5,000** iterations according to their saved parameter files, so they are not directly comparable to the 3,000-iteration runs.

The three lowest recorded validation losses are:

1. **Hidden size 256, learning rate 0.0005:** 1.368786
2. **Hidden size 256, learning rate 0.0001:** 1.371095
3. **Hidden size 256, learning rate 0.001:** 1.373872

All three top runs use hidden size 256. On this held-out validation split, the smaller 256-unit LSTM generalized better than the larger 512- and 1024-unit versions: it achieved lower validation loss, suggesting that the larger models did not improve next-character prediction on unseen corpus text. The training and validation loss plots also show a larger gap for bigger models, which is consistent with more overfitting as model size increases in these experiments. This is evidence about this dataset and split, not a guarantee that 256 units will always be best. Validation loss measures character prediction, not musical quality or ABC validity.

The notebook's `sorted_models` table contains all checkpoint results. `models/<run-name>/best_model.pt` stores the best validation checkpoint for a run, `params.json` records that run's settings, and `loss.png` shows its training and validation curves. The notebook currently selects `sorted_models.iloc[1]` for generation, which is the **second** row after sorting by validation loss; change the index to `0` to select the top-ranked run. `all_params.json` contains 20 configurations, while the notebook's input file `params_list.json` is currently empty (`[]`). Copy the desired configurations into `params_list.json` before running training; otherwise the training loop is skipped. Existing checkpoints can be used without retraining if their dimensions match the notebook's vocabulary and model.

## ABC validity and improving generations

The model predicts one character at a time and is not constrained by ABC's grammar. It can therefore produce incomplete headers, malformed notation, or tunes that `abc2midi` cannot parse. A generated sample may also end before a tune is complete. This is expected for unconstrained character-level generation, especially with limited training data or training time.

Possible ways to improve the result include:

- Train longer and/or add more clean ABC tunes so the model sees more valid notation patterns.
- Start from a complete, well-formed ABC header and generate enough characters for the tune to finish.
- Tune sampling randomness (for example, use a temperature or top-k sampling) to reduce unlikely character sequences while keeping some variety.
- Validate generated text with `abc2midi` and discard or resample outputs that fail; keep only complete tunes that pass validation.
- For stronger guarantees, constrain decoding with ABC syntax rules or use a parser-aware generation method.

Audio conversion requires both `abc2midi` and `timidity`. If conversion fails, first check that these commands are installed and available on `PATH`, then inspect the generated `.abc` file for syntax or completion problems.

## Notes

- Run the notebook with the project root as its working directory; its data, model, and output paths are project-relative in the helper module, while some notebook paths use the current working directory.
- `src/music/util.py` also contains a few general-purpose plotting helpers that are not central to music generation.
