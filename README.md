# Shakespearean Text Generation with Transformer

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Run on Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/evangelosgakias/transformer-nlp-tensorflow)

> **A comprehensive, from-scratch implementation of the Transformer architecture for generating Shakespearean text, built with TensorFlow and designed for deep learning research and creative AI.**

---

## 🚀 Project Overview

This project implements a complete Transformer model from scratch for word-level language modeling and text generation in the style of William Shakespeare. It is designed as both an educational resource and a showcase of advanced deep learning engineering, following the "Attention Is All You Need" paper (Vaswani et al., 2017).

- **Full Transformer architecture** (encoder-decoder, multi-head attention, positional encoding, etc.)
- **Word-level modeling** for rich, context-aware text generation
- **Custom training pipeline** with data augmentation, label smoothing, and advanced learning rate scheduling
- **Extensive documentation and code comments** for learning and reproducibility

---

## 📂 Directory Structure

```
├── transformer.ipynb      # Main Jupyter notebook (full implementation & experiments)
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
├── data/                  # Processed data, vocabularies, and numpy arrays (auto-generated)
├── checkpoints/           # Model checkpoints (auto-generated)
├── saved_model/           # Saved trained models (auto-generated)
└── LICENSE                # MIT License
```

---

## 🧑‍💻 Quick Start

### 1. **Clone the Repository**

```bash
git clone https://github.com/EvanGks/shakespeare-text-generation-transformer.git
cd shakespeare-text-generation-transformer
```

### 2. **Set Up the Environment**

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. **Run the Notebook**

Open `transformer.ipynb` in Jupyter and run all cells sequentially:

```bash
jupyter notebook transformer.ipynb
```

---

## 📊 Features

- **Complete Transformer implementation** (no high-level Keras API for core logic)
- **Word-level language modeling** for richer context
- **Custom learning rate scheduler** with warmup
- **Advanced text generation** (temperature, top-k sampling)
- **Data augmentation** (word dropout, random swaps)
- **Label smoothing** for better generalization
- **Early stopping & checkpointing**
- **Comprehensive evaluation** (accuracy, perplexity, qualitative samples)
- **Visualization of training metrics**
- **Well-documented, modular code**

---

## 🏗️ Model Architecture

- **Encoder-Decoder Transformer**
- **Multi-head self-attention**
- **Position-wise feed-forward networks**
- **Sinusoidal positional encoding**
- **Dropout and label smoothing**

**Key Hyperparameters:**
- Layers: 3
- Embedding dim: 192
- Attention heads: 6
- Feed-forward dim: 768
- Dropout: 0.3

> **Architecture Diagram:**
>
>```
>+-------------------------------------------------------------+
>|                    TRANSFORMER ARCHITECTURE                 |
>|-------------------------------------------------------------|
>|                                                             |
>|  Input Sequence (word indices)                              |
>|         |                                                   |
>|         v                                                   |
>|  +-------------------+                                      |
>|  | Embedding Layer   |  (dim: 192)                          |
>|  +-------------------+                                      |
>|         |                                                   |
>|  +---------------------------+                              |
>|  | Positional Encoding       |  (sinusoidal, max len: 5000) |
>|  +---------------------------+                              |
>|         |                                                   |
>|         v                                                   |
>|  +-------------------------------------------------------+  |
>|  |                   ENCODER STACK (3 layers)            |  |
>|  |  - Multi-Head Self-Attention (6 heads, dim: 192)      |  |
>|  |  - Feed-Forward Network (dim: 768)                    |  |
>|  |  - Dropout: 0.3, LayerNorm, Residual                  |  |
>|  +-------------------------------------------------------+  |
>|         |                                                   |
>|         v                                                   |
>|  +-------------------------------------------------------+  |
>|  |                   DECODER STACK (3 layers)            |  |
>|  |  - Masked Multi-Head Self-Attention (6 heads, 192)    |  |
>|  |  - Encoder-Decoder Attention (6 heads, 192)           |  |
>|  |  - Feed-Forward Network (dim: 768)                    |  |
>|  |  - Dropout: 0.3, LayerNorm, Residual                  |  |
>|  +-------------------------------------------------------+  |
>|         |                                                   |
>|         v                                                   |
>|  +-------------------+                                      |
>|  | Linear Projection |  (to vocab size)                     |
>|  +-------------------+                                      |
>|         |                                                   |
>|         v                                                   |
>|  Output Sequence (predicted word indices)                   |
>+-------------------------------------------------------------+
>```
>
> *A high-level overview of the Transformer model used in this project. Key hyperparameters: 3 layers, 192 embedding dim, 6 attention heads, 768 feed-forward dim, 0.3 dropout.*

---

## 📈 Results

- **Training accuracy:** ~92%
- **Validation accuracy:** ~86%
- **Coherent Shakespearean text generation**
- **Captures character dialogue and style**

**Example Outputs:**
```
HAMLET: To be, or not to be, that is the question
ROMEO: But soft, what light through yonder window breaks
MACBETH: Is this a dagger which I see before me
```

---

## 📥 Dataset

- **Source:** [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) (by Andrej Karpathy)
- ~1 million characters from Shakespeare's plays and sonnets
- Includes dialogue, stage directions, and scene descriptions

---

## 🛠️ Usage Example

Generate text after training:

```python
from transformer import generate_text  # If you modularize the code

sample = generate_text(
    model=word_transformer,
    start_string="HAMLET:",
    word2idx=word2idx,
    idx2word=idx2word,
    generation_length=30,
    temperature=0.7
)
print(sample)
```

---

## 🧩 Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

- **Fork** the repository
- **Create a feature branch** (`git checkout -b feature/your-feature`)
- **Commit your changes** (`git commit -m 'feat: add new feature'`)
- **Push to the branch** (`git push origin feature/your-feature`)
- **Open a Pull Request**

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Tiny Shakespeare dataset (Karpathy)](https://github.com/karpathy/char-rnn)
- Open-source deep learning community

---

## 🌟 Why This Project?

This project demonstrates advanced deep learning engineering, from data preprocessing to custom model implementation and evaluation. It is designed to showcase:

- **Mastery of modern NLP architectures**
- **Ability to build complex models from scratch**
- **Best practices in code, documentation, and reproducibility**
- **A passion for both research and creative AI**

---

## 📈 Live Results

You can view the notebook with all outputs and results on Kaggle:
[https://www.kaggle.com/code/evangelosgakias/transformer-nlp-tensorflow](https://www.kaggle.com/code/evangelosgakias/transformer-nlp-tensorflow)

---

## 📬 Contact
For questions or feedback, please reach out via:

- **GitHub:** [EvanGks](https://github.com/EvanGks)
- **X (Twitter):** [@Evan6471133782](https://x.com/Evan6471133782)
- **LinkedIn:** [Evangelos Gakias](https://www.linkedin.com/in/evangelos-gakias-346a9072)
- **Kaggle:** [evangelosgakias](https://www.kaggle.com/evangelosgakias)
- **Email:** [evangks88@gmail.com](mailto:evangks88@gmail.com)

---

Happy Coding! 🚀
