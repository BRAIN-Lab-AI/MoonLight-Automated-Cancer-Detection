# Moonlight: Enhanced Deep Learning Models for Breast Cancer Detection Using Histopathology Images

## Introduction

**Moonlight** is an advanced deep learning framework designed to improve the accuracy, robustness, and interpretability of breast cancer diagnosis from histopathological images. The project introduces modular CNN architectures, novel loss functions, enhanced data augmentation strategies, and explainability features through Grad-CAM++. 

Moonlight aims to support AI-assisted diagnostics with clinically relevant outcomes and provides researchers with a configurable pipeline for experimentation using the BreaKHis dataset.

## Project Metadata

### Authors
- **Author:** Hassan Jawad Al-Dahneen  
- **Supervisor:** Dr. Muzammil Behzad  
- **Affiliation:** King Fahd University of Petroleum and Minerals (KFUPM)

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Moonlight Project Report](Moonlight__Enhanced_Deep_Learning_Models_for_Breast_Cancer_Detection_Using_Histopathology_Image.pdf)

### Reference Paper
- [Hybrid Deep Learning for Breast Cancer Histopathology](https://pmc.ncbi.nlm.nih.gov/articles/PMC11191493/)

### Reference Dataset
- [BreaKHis Dataset on Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis?resource=download)

### Reference Code Repository
- [GitHub Repo](https://github.com/mrdvince/breast_cancer_detection)


## Project Technicalities

## Terminologies

This section explains key technical and domain-specific terms used in the Moonlight project, spanning both deep learning concepts and histopathological breast cancer analysis.

### Deep Learning & Technical Terminologies

- **SimpleCNN:** A basic convolutional neural network used as a baseline model. It contains three convolutional layers, adaptive pooling, and a fully connected classifier. It helps establish reference performance for evaluating enhanced models.
- **ResidualModel:** A CNN architecture that introduces *residual connections* or skip links to allow gradients to flow directly through the network. This improves training stability, especially in deep architectures.
- **EfficientConvModel:** A lightweight architecture that uses **depthwise separable convolutions** to reduce computational load while preserving accuracy. Inspired by MobileNet design principles.
- **UNetClassifier:** A classifier variant of the U-Net architecture. Originally developed for biomedical segmentation, it combines downsampling and upsampling paths to retain spatial features — ideal for capturing microscopic patterns in histology.
- **DenseNet121:** A densely connected convolutional network where each layer receives feature maps from all preceding layers. Enhances feature reuse and gradient propagation.
- **Cross Entropy Loss:** Measures the performance of a classification model by penalizing incorrect predictions based on predicted probabilities.
- **Focal Loss:** Addresses class imbalance by down-weighting well-classified examples, making the model focus on harder, minority-class samples.
- **Perceptual Loss:** Computes the Mean Squared Error (MSE) between predicted softmax probabilities and one-hot encoded labels. Enhances perceptual similarity learning.
- **Composite Loss:** Combines Cross Entropy and Perceptual Loss to improve both class discrimination and semantic alignment.
- **Basic Augmentation:** Includes resizing, normalization, and horizontal flipping. Helps improve generalization with minimal transformation overhead.
- **Advanced Augmentation:** Includes random resized cropping, flipping, rotation, and color jittering. Enhances robustness by simulating real-world variability in histological slide appearance.
- **Grad-CAM / Grad-CAM++:** Gradient-based visual explanation tools that highlight regions of an image most influential to the model's decision. Grad-CAM++ offers better localization and finer heatmaps.
- **AMP (Automatic Mixed Precision):** Speeds up training and reduces memory usage by mixing 16-bit and 32-bit floating-point calculations.
- **StepLR Scheduler:** Decreases the learning rate at regular intervals to promote stable convergence.
- **Early Stopping:** Halts training if the validation loss does not improve after a specified number of epochs, preventing overfitting.

### Medical & Domain-Specific Terminologies

- **Breast Cancer:** A malignant tumor that originates from breast tissue. Early detection through image-based diagnostics significantly improves treatment outcomes.
- **Histopathology:** The microscopic examination of tissue samples to identify signs of disease. In this project, histopathological slides of breast tissue are used for automated classification.
- **Benign Tumor:** A non-cancerous growth that does not invade nearby tissues or spread. Examples in the BreaKHis dataset include fibroadenomas and phyllodes tumors.
- **Malignant Tumor:** A cancerous mass with invasive and metastatic potential. Includes types such as ductal carcinoma, lobular carcinoma, and papillary carcinoma in the BreaKHis dataset.
- **Hyperchromatic Nuclei:** Dark-stained cell nuclei often found in malignant tissues due to increased DNA content — a critical visual marker for pathologists and deep learning models.
- **Nucleus-to-Cytoplasm Ratio (N:C Ratio):** A higher N:C ratio is commonly observed in malignant cells. Deep learning models can implicitly learn such features during training.
- **Tissue Architecture:** Describes the spatial organization of cells and structures in a tissue sample. Cancer often causes architectural distortion, which can be learned by U-Net and residual architectures.
- **BreaKHis Dataset:** A histopathological image dataset containing 7,909 annotated images of benign and malignant breast tissue at multiple magnifications (40X, 100X, 200X, 400X).
- **Magnification (400X):** Refers to the level of image zoom. The Moonlight project primarily works at the 400X magnification level to capture high-resolution cellular features.
- **H&E Stain (Hematoxylin & Eosin):** Common staining method used in histology. Hematoxylin stains nuclei blue; eosin stains the cytoplasm pink. The RGB format preserves these color cues, which are critical for accurate classification.
- **Pathologist:** A medical expert who interprets histological slides. Moonlight aims to support their diagnosis by offering AI-generated second opinions with visual evidence (heatmaps).

### Problem Statements
1. **Architectural Limitations:**  
   Existing CNN models lack the depth and flexibility needed to capture the subtle tissue patterns and cellular variations in histopathological images, especially across different tumor subtypes and magnifications.
2. **Loss Function Sensitivity and Class Imbalance:**  
   Conventional loss functions such as Cross Entropy do not adequately reflect the domain-specific challenges in medical imaging, including class imbalance and the high cost of false negatives.
3. **Insufficient Data Augmentation:**  
   Basic image augmentations do not simulate the complex variability found in real-world histological slides, such as lighting inconsistencies, staining variations, or tissue deformation.
4. **Lack of Interpretability:**  
   Without model transparency, predictions made by deep learning models are difficult for pathologists to trust, limiting clinical adoption.
5. **Grayscale Input and Color Loss:**  
   Many existing models operate on grayscale images, discarding vital color-based diagnostic cues like nuclear staining intensity and cytoplasmic texture.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
