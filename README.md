# AI vs Real Image Detector

## Overview

The rapid rise of generative AI has made it increasingly difficult to distinguish between authentic photographs and AI-generated content. This project presents a deep learning-based image classification system that identifies whether an image is **Real** or **AI-Generated** using transfer learning with a pretrained ResNet-18 architecture.

The model is trained on the **CIFAKE dataset**, a large collection of real and synthetic images, and achieves over **96% validation accuracy**, demonstrating strong performance in detecting AI-generated visual content.

---

## Features

* Binary image classification:

  * REAL
  * FAKE (AI-Generated)
* Transfer learning using pretrained ResNet-18
* GPU-accelerated training with CUDA support
* Image augmentation and normalization
* Training and validation performance tracking
* High accuracy with minimal training epochs

---

## Dataset

This project uses the **CIFAKE: Real and AI-Generated Synthetic Images Dataset** available on Kaggle.

Dataset Structure:

```
train/
├── REAL/
└── FAKE/

test/
├── REAL/
└── FAKE/
```

The dataset contains a balanced collection of authentic and synthetic images designed specifically for AI-image detection research.

---

## Model Architecture

The project utilizes:

* **ResNet-18** pretrained on ImageNet
* Modified final fully connected layer
* Output classes: 2

```python
model.fc = nn.Linear(model.fc.in_features, 2)
```

Transfer learning enables the model to leverage previously learned visual features while adapting to the AI image detection task.

---

## Data Preprocessing

Images undergo the following transformations:

* Resize to 224 × 224
* Random Horizontal Flip
* Tensor Conversion
* Normalization

```python
transforms.Resize((224,224))
transforms.RandomHorizontalFlip()
transforms.ToTensor()
transforms.Normalize([0.5,0.5,0.5],
                     [0.5,0.5,0.5])
```

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Model         | ResNet-18        |
| Optimizer     | Adam             |
| Learning Rate | 0.001            |
| Loss Function | CrossEntropyLoss |
| Batch Size    | 32               |
| Epochs        | 5                |
| Device        | CUDA / GPU       |

---

## Results

| Epoch | Training Accuracy | Validation Accuracy |
| ----- | ----------------- | ------------------- |
| 1     | 91.84%            | 91.43%              |
| 2     | 94.27%            | 95.19%              |
| 3     | 95.31%            | 95.87%              |
| 4     | 96.00%            | 96.46%              |
| 5     | 96.46%            | **96.61%**          |

### Final Performance

* Training Accuracy: **96.46%**
* Validation Accuracy: **96.61%**
* Validation Loss: **0.0895**

The model demonstrates excellent generalization with minimal overfitting.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-image-detector.git
cd ai-image-detector
```

Install dependencies:

```bash
pip install torch torchvision tqdm kaggle matplotlib
```

---

## Running the Project

### 1. Download Dataset

```bash
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
```

### 2. Extract Dataset

```bash
unzip cifake-real-and-ai-generated-synthetic-images.zip
```

### 3. Train the Model

Run the notebook or training script:

```bash
python train.py
```

---

## Applications

* AI-generated content detection
* Digital media verification
* Social media authenticity checks
* Academic research
* Misinformation prevention
* Content moderation systems

---

## Future Improvements

* Support for multiple generative models
* Explainable AI visualizations (Grad-CAM)
* Vision Transformer (ViT) implementation
* Real-time web application deployment
* Confidence score analysis
* Multi-class synthetic image classification

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* CUDA
* Kaggle API
* ResNet-18
* CIFAKE Dataset

---

## Disclaimer

This project is intended for educational and research purposes. As image generation technology continues to evolve, detection systems should be continuously retrained and evaluated against newer generative models.

---

## Author

Developed as a deep learning and computer vision project exploring the detection of AI-generated imagery using modern transfer learning techniques.
