# Falcons — AI-Powered Navigation-Error Recovery for Wafer Inspection

***Overview***

This project implements an AI-powered computer vision system for recovering navigation errors in semiconductor wafer inspection tools.

The system takes a **Reference Image** representing the target site and a **Search Image** containing the larger wafer region, and predicts the **center `(x, y)` coordinate** of the reference pattern within the search image.

Instead of relying on classical pixel-level template matching, the solution uses a **learned Siamese cross-correlation network** that learns discriminative visual features and performs localization through a correlation heatmap.

The approach is designed for highly repetitive semiconductor structures such as **DRAM and FinFET layouts**, where conventional template matching can produce multiple visually similar matches.

---

***Key Features***

* Learned **Siamese CNN** with shared-weight feature extraction
* Feature-level **cross-correlation** for reference-to-search matching
* **Soft-argmax** coordinate prediction for differentiable localization
* Coordinate + heatmap based training objective
* Designed for repetitive **DRAM / FinFET semiconductor patterns**
* Lightweight deep learning model
* Standalone inference script requiring only a reference image and search image
* Synthetic dataset generator with recorded ground-truth coordinates
* Trained PyTorch model weights included
* Complete training notebook included for reproducibility

---

***Model Summary***

| **Item**                  | **Value**                                 |
| ------------------------- | ----------------------------------------- |
| Architecture              | Siamese CNN + Cross-Correlation           |
| Training Approach         | End-to-End Deep Learning                  |
| Framework                 | PyTorch                                   |
| Localization Method       | Learned Feature Correlation + Soft-Argmax |
| Input                     | Reference Image + Search Image            |
| Search Processing Size    | 512 × 512                                 |
| Reference Processing Size | 51 × 51                                   |
| Model Parameters          | 60,656                                    |
| Model Weights             | `model_weights.pt`                        |
| Output                    | Predicted center `(x, y)`                 |
| Target Application        | Wafer Inspection Navigation Recovery      |

---

***How It Works***

The localization pipeline consists of four main stages:

**1. Shared Feature Extraction**

The Reference and Search images are passed through the same CNN backbone. Shared weights allow the network to learn a common feature representation for both images.

**2. Feature Cross-Correlation**

The reference feature representation is correlated across the search feature map to generate a spatial matching response.

**3. Localization Heatmap**

The correlation response is upsampled to produce a heatmap representing the likelihood of the target location.

**4. Coordinate Prediction**

A differentiable soft-argmax converts the heatmap into the final predicted center coordinate:

```text
(x, y)
```


***Dataset Summary***

The model was trained using synthetically generated semiconductor image pairs consisting of a Reference Image and a Search Image.

| **Item**                      | **Value**                  |
| ----------------------------- | -------------------------- |
| Total image pairs             | 750                        |
| Training samples              | 525                        |
| Validation samples            | 105                        |
| Test samples                  | 120                        |
| Reference image               | 1000 × 1000 px             |
| Search image                  | 1000 × 1000 px             |
| Reference footprint in Search | 100 × 100 px               |
| Target                        | Center `(x, y)` coordinate |
| Semiconductor structure       | FinFET                     |

The dataset generation process records the exact target bounding box and center coordinate for every generated reference-search pair.

---

***Repository Structure***

AI-Powered-Navigation-Error-Recovery
├── README.md
├── inference.py
├── dataset_generator.py
├── model_weights.pt
├── Falcons_model_train.ipynb
├── requirements.txt
├── MODEL_CARD.md
│
├── references/
│   ├── references.md
│   └── GENERATOR_PROVENANCE.md
│
└── examples/
    └── README.md
```

---

***How to Run***

**1. Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/Falcons.git
cd Falcons
```

**2. Install Requirements**

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

**3. Run Localization Inference**

The inference script accepts two image paths:

```bash
python inference.py \
    --reference path/to/reference.png \
    --search path/to/search.png
```

The trained model weights are loaded automatically from:

```text
model_weights.pt
```

**Output:**

```text
x,y
```

Example:

```text
498.68,485.15
```

The output represents the predicted center of the reference pattern inside the search image.

---

***Generate a Dataset***

The standalone dataset generator can generate synthetic Reference + Search image pairs and their corresponding ground-truth coordinates.

For FinFET:

```bash
python dataset_generator.py \
    --architecture FinFET \
    --pairs 20 \
    --output generated_dataset
```

For DRAM:

```bash
python dataset_generator.py \
    --architecture DRAM \
    --pairs 20 \
    --output generated_dataset
```

The generated directory contains:

```text
generated_dataset/
│
├── search/
├── reference/
└── ground_truth.csv
```

The CSV records the target bounding box and center coordinate for every generated pair.

---

***Train the Model***

The complete training process is provided in:

```text
Falcons_model_train.ipynb
```

The notebook contains:

* Dataset loading
* Image preprocessing
* Training augmentation
* Siamese CNN definition
* Feature cross-correlation
* Heatmap generation
* Soft-argmax localization
* Training and validation
* Model checkpoint saving
* Test evaluation

The notebook can be opened directly in **Google Colab** for training.

---

***Results***

The submitted model is a lightweight localization network with **60,656 trainable parameters** and approximately **248 KB** of model weights.


---

***Why This Approach***

Classical pixel-level template matching can struggle when semiconductor layouts contain many repeated structures with similar appearance.

Falcons instead learns feature representations from the Reference and Search images and performs matching in the learned feature space.

This enables the model to use structural information rather than relying only on raw pixel similarity.

The system is also **one-shot at inference time**: a new reference image can be supplied directly without retraining the network for that specific target.

---

***Reproducibility***

The repository includes all major components required to reproduce and evaluate the solution:

* Dataset generator
* Training notebook
* Trained model weights
* Standalone inference script
* Python dependencies
* Supporting references

The inference script is designed to run without manual modification and accepts the Reference and Search image paths directly as command-line arguments.

---

**Team Falcons**

AI-Powered Navigation-Error Recovery for Wafer Inspection Tools
