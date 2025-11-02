# Vision Transformer (ViT) on Fashion-MNIST

**Name:** Rishikesh Kumar  
**Roll No:** 22052747  
**Course:** Deep Learning / Advanced Neural Networks  
**Assignment:** Exploring Transformers and Vision Transformers (ViT)

---

## Project Overview
This project implements a Vision Transformer (ViT) from scratch using PyTorch on the Fashion-MNIST dataset.  
It follows the assignment structure including:
- Patch embedding and multi-head self-attention  
- Roll-number-based parameterization  
- Training and testing pipeline  
- CNN comparison for performance  
- Attention visualization for analysis  

---

## Setup Instructions

### 1. Open Notebook
Open the `.ipynb` file in Jupyter Notebook or Google Colab.

### 2. Install Dependencies
```bash
pip install torch torchvision tqdm scikit-learn matplotlib opencv-python
3. Run the Notebook
Execute all cells in order:

Import libraries and set parameters

Dataset download and preprocessing

ViT model training and testing

Plot accuracy and confusion matrix

CNN comparison (bonus)

Attention visualization (bonus)

Dataset
Dataset: Fashion-MNIST
TorchVision Link: FashionMNIST

The dataset downloads automatically when you run:

python
Copy code
datasets.FashionMNIST(root='./data', download=True)
Classes:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Roll-Based Hyperparameters (Roll No: 22052747 → seed = 47)
Parameter	Formula	Value
hidden_dim	128 + (seed % 5) × 32	192
num_heads	4 + (seed % 3)	6
patch_size	8 + (seed % 4) × 2	14
epochs	10 + (seed % 5)	12

Model Details
Vision Transformer (ViT)
Custom patch embedding (Conv2D)

Multi-Head Self-Attention (no pretraining)

4 Encoder layers, GELU activation

Linear classification head (10 classes)

CNN (for Comparison)
3 Convolution layers + MaxPooling

2 Fully Connected layers

ReLU activations

Results Summary
Model	Test Accuracy	Notes
Vision Transformer (ViT)	~84–86%	Custom implementation
Simple CNN	~87–89%	Performs slightly better

Bonus (Implemented)
✅ ViT vs CNN comparison

✅ Attention visualization on ViT layers

🎥 Demo video showing predictions (demo_ViT_22052747.mp4)

Files Included
File	Description
ViT_22052747.ipynb	Main code notebook
report_22052747.pdf	Theory + analysis report
demo_ViT_22052747.mp4	Short demo video
README.md	Project documentation

Reproducibility
To re-use the trained model:

python
Copy code
model.load_state_dict(torch.load('vit_model_22052747.pth'))
model.eval()
Or simply re-run all cells for new training.

References
Vaswani et al., Attention Is All You Need, NeurIPS 2017

Dosovitskiy et al., An Image Is Worth 16×16 Words, ICLR 2021

PyTorch Vision Docs – https://pytorch.org/vision/stable/

Author: Rishikesh Kumar
Roll No: 22052747
Institution: KIIT University
Submission: Vision Transformer Assignment