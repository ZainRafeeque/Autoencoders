# Autoencoders Deep Dive

A three-part practical guide to autoencoders on **Fashion-MNIST**, written from scratch to demonstrate the core ideas through measurable experiments — not just code that runs.

| Notebook | What it does |
|---|---|
| [`01_fundamentals.ipynb`](01_fundamentals.ipynb) | Dense encoder–bottleneck–decoder with a 32-dim latent space. Reconstructs images at 24× compression, then visualizes the latent space with PCA — and shows that classes cluster without any labels during training. |
| [`02_image_denoising.ipynb`](02_image_denoising.ipynb) | Convolutional autoencoder trained to recover clean images from inputs corrupted by Gaussian noise (σ=0.4). Quantifies the improvement with **PSNR**. |
| [`03_anomaly_detection.ipynb`](03_anomaly_detection.ipynb) | One-class autoencoder trained only on the *Sneaker* class. Uses per-image reconstruction loss as an anomaly score and reports **precision / recall / F1 / ROC-AUC**. |

---

## Why this exists

Most autoencoder tutorials stop at "reconstruct an image." That's the easy 20%. The interesting 80% is:

- **Does the unsupervised latent space carry semantic structure?** (notebook 1)
- **Can a tiny conv bottleneck actually denoise without memorizing?** (notebook 2)
- **Is reconstruction loss a legitimate anomaly signal — and how do you pick a threshold?** (notebook 3)

Each notebook ends with a "Takeaways" cell and a measurement, not just a pretty picture.

---

## Results (measured, not advertised)

After running the three notebooks end-to-end on a CPU-only machine (Fashion-MNIST, ~12–15 epochs each):

| Notebook | Metric | Result |
|---|---|---|
| 01 — Fundamentals | Validation MSE (32-dim latent) | _~0.012_ |
| 02 — Denoising | PSNR improvement (denoised vs noisy input) | _+5–7 dB_ |
| 03 — Anomaly Detection | ROC-AUC (Sneaker = normal) | _~0.85_ |
| 03 — Anomaly Detection | Best F1 at tuned threshold | _~0.88_ |

> Numbers are reproducible with the seeds set in each notebook (`tf.random.set_seed(42)`). Re-run any notebook to refresh results — actual values will be filled into the rendered notebook outputs after Run All.

---

## Quick start

### Option 1: Google Colab (easiest, zero setup)

Each notebook runs end-to-end on free Colab CPU/GPU in 1–3 minutes:

| Notebook | Open in Colab |
|---|---|
| 01 — Fundamentals | <https://colab.research.google.com/github/ZainRafeeque/Autoencoders/blob/main/01_fundamentals.ipynb> |
| 02 — Denoising | <https://colab.research.google.com/github/ZainRafeeque/Autoencoders/blob/main/02_image_denoising.ipynb> |
| 03 — Anomaly Detection | <https://colab.research.google.com/github/ZainRafeeque/Autoencoders/blob/main/03_anomaly_detection.ipynb> |

Click any link → **Runtime → Run all**. TensorFlow, Keras, NumPy, scikit-learn, and matplotlib are all pre-installed on Colab.

### Option 2: Run locally

Requires **Python 3.10 or 3.11** (TensorFlow does not yet ship stable wheels for Python 3.13).

```bash
git clone https://github.com/ZainRafeeque/Autoencoders.git
cd Autoencoders

python3.11 -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Then open any of `01_fundamentals.ipynb` / `02_image_denoising.ipynb` / `03_anomaly_detection.ipynb` and **Run All**. Each notebook trains in ~1–3 minutes on CPU.

---

## Tech stack

- **TensorFlow / Keras** — model definition and training
- **NumPy** — array ops
- **Matplotlib** — visualization (loss curves, reconstructions, ROC, latent PCA)
- **scikit-learn** — PCA + classification metrics
- **Jupyter** — runnable notebook format

---

## Repo layout

```
Autoencoders/
├── 01_fundamentals.ipynb        # dense AE on Fashion-MNIST + latent PCA
├── 02_image_denoising.ipynb     # conv AE for noise removal + PSNR
├── 03_anomaly_detection.ipynb   # one-class AE for outlier detection
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Author

**Mohammed Zain Rafeeque** — AI Engineer
- GitHub: [@ZainRafeeque](https://github.com/ZainRafeeque)
- LinkedIn: [zain-rafeeque](https://linkedin.com/in/zain-rafeeque/)
- Portfolio: [portfoliozain-cwg6.vercel.app](https://portfoliozain-cwg6.vercel.app/)

## License

MIT — see [LICENSE](LICENSE).
