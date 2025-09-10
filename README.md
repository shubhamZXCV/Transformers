# Neural Machine Translation with Transformers

This project implements a Transformer-based sequence-to-sequence model for **machine translation** using PyTorch.  
It supports **multiple positional encoding strategies** (RoPE, Relative Bias, Absolute), custom tokenization, and multiple decoding strategies (Greedy, Beam Search, Top-k sampling).

---

## 📂 Project Structure

```

.
├── config.py               # Configuration utilities
├── dataset.py              # Custom bilingual dataset & dataloader
├── model.py                # Transformer architecture
├── train.py                # Training loop (main entrypoint)
├── infer.py                # Inference & decoding strategies
├── test.py           # BLEU score evaluation
├── runs/                   # TensorBoard logs
├── weights\_\*               # Saved checkpoints
└── dataset.pkl             # Preprocessed dataset (train/val/test)

````

---

## ⚙️ Setup

1. **Clone & enter repo**
   ```bash
   git clone git@github.com:shubhamZXCV/Transformers.git
   cd Transformers
    ```

2. **Create environment**

   ```bash
   conda create -n nmt python=3.9 -y
   conda activate nmt
   ```

3. **Install dependencies**

   ```bash
   pip install
   ```

   Key packages:

   * `torch`, `torchtext`, `torchmetrics`
   * `datasets`, `tokenizers`
   * `tensorboard`, `tqdm`, `sacrebleu`

---

## ⚙️ Configuration (`config.py`)

All training & inference parameters are set in **`config.py`**.
Here’s the default config:

```python
from pathlib import Path

def get_config():
    return {
        "batch_size": 24,
        "num_epochs": 15,
        "lr": 10**-4,
        "seq_len": 270,
        "d_model": 512,
        "datasource": "EUbookshop",
        "lang_src": "fi",
        "lang_tgt": "en",
        "model_folder": "weights_relbias",
        "model_basename": "tmodel_",
        "preload": None,   # "latest" or epoch number as string (e.g. "07")
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_relbias",
        "positional_encoding": "relative_bias",  # Options: "rope", "relative_bias", "absolute"
        "decoding_strategy": "beam"              # Options: "greed", "beam", "topk"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
```

🔑 **Key parameters to adjust:**

* `batch_size`, `num_epochs`, `lr` → training hyperparams
* `seq_len`, `d_model` → model capacity
* `lang_src`, `lang_tgt` → source & target language codes
* `positional_encoding` → choose from `"rope"`, `"relative_bias"`, `"absolute"`
* `decoding_strategy` → choose from `"greed"`, `"beam"`, `"topk"`
* `preload`:

  * `None` → train from scratch
  * `"latest"` → resume from last checkpoint
  * `"07"` → load specific checkpoint

---

## 📊 Dataset

The dataset is preprocessed and stored in `dataset.pkl`.
It contains:

* Train split
* Validation split
* Test split

Tokenizers are automatically built (and saved) in JSON files:

```
tokenizer_{src}.json
tokenizer_{tgt}.json
```

---

## 🚀 Training

Run training with:

```bash
python train.py
```



---

## 💾 Checkpointing

Models are saved under:

```
{datasource}_{model_folder}/tmodel_{epoch}.pt
```

* To **resume training**, set in `config.py`:

  ```python
  "preload": "latest"
  ```

---

## 📈 Monitoring with TensorBoard

Logs are written under `runs/`.
Example config keys:

* `"experiment_name": "runs/tmodel_rope"`
* `"experiment_name": "runs/tmodel_relbias"`

Run TensorBoard:

```bash
tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006).

---

## 📝 Evaluation

Use `test.py` (or the BLEU code inside `calculate_bleu`) to compute corpus-level BLEU:

```bash
python test.py
```

Decoding strategies supported:

* Greedy
* Beam Search
* Top-k Sampling



---

Got it ✅ You want a clear **section in the README** that documents what goes inside your log files (`train_*_logs.txt` and `test_*_logs.txt`) so that anyone running your repo knows what to expect.

Here’s a well-structured addition you can drop into the README:

---

## 🗂️ Log Files

During training and evaluation, logs are written to text files for later inspection.

### 📒 Training Logs

* Files:

  * `train_rope_logs.txt`
  * `train_relbias_logs.txt`

* Contents per epoch:

  * **Training Loss**
  * **Validation Loss**
  * **Qualitative Samples** (printed after each epoch):

    ```
    INPUT   : <source sentence>
    TARGET  : <ground truth translation>
    PREDICT : <model translation>
    ```

* Purpose:

  * Track convergence of loss values
  * Monitor qualitative progress of translations as training advances

---

### 🧪 Testing Logs

* Files:

  * `test_rope_logs.txt`
  * `test_relbias_logs.txt`

* Contents:

  * **BLEU Score** for each decoding strategy:

    * Greedy
    * Beam Search
    * Top-k Sampling
  * **Qualitative Examples**:

    ```
    INPUT   : <source sentence>
    TARGET  : <ground truth translation>
    PREDICT : <model translation>
    ```

* Purpose:

  * Compare decoding strategies quantitatively (BLEU) and qualitatively (sample translations)
  * Identify trade-offs between different decoding strategies

---




### Link to the model weights
[click me to get the model weights](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/shubham_goel_students_iiit_ac_in/El68FSetP3lPsHMTgwqCw9IBuMW_pnwJhAhAaTC7vKJBDg?e=JAt6QA)