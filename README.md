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
1. **Domain Misalignment of Pretrained Architectures:** Pretrained CNNs like DenseNet121 are deep and powerful, but they were originally designed for natural image datasets (e.g., ImageNet). As such, they may not be optimally structured to capture domain-specific features in histopathological images — such as nuclear morphology, tissue architecture, and staining patterns — without significant fine-tuning or architectural adaptation.
2. **Insufficient Data Augmentation:** Basic image augmentations do not simulate the complex variability found in real-world histological slides, such as lighting inconsistencies, staining variations, or tissue deformation.
3. **Lack of Interpretability:** Without model transparency, predictions made by deep learning models are difficult for pathologists to trust, limiting clinical adoption.
4. **Grayscale Input and Color Loss:** Many existing models operate on grayscale images, discarding vital color-based diagnostic cues like nuclear staining intensity and cytoplasmic texture.


### Loopholes or Research Areas
- Over-reliance on one architecture (e.g., DenseNet121) without exploring efficiency vs. performance tradeoffs in custom-designed models.
- Absence of hybrid loss functions that combine class-wise accuracy and semantic alignment.
- Limited data augmentation beyond basic flipping or resizing.
- Few pipelines offer clinical interpretability via explainable AI tools.
- Lack of modular, reusable deep learning frameworks for histopathology analysis.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

| Problem | Proposed Solution |
|--------|--------------------|
| **Model Inflexibility & Domain Misalignment** | **Optimized Architecture**<br>Introduce and benchmark modular CNNs (e.g., ResidualModel, EfficientConvModel, UNetClassifier) that are tailored for histopathological feature extraction. |
| **Overfitting & Poor Generalization** | **Enhanced Data Augmentation**<br>Apply advanced augmentations (cropping, rotation, jittering) to reflect clinical variability and improve robustness. |
| **Color Information Loss in Grayscale Input** | **RGB Image Utilization**<br>Train models using RGB histopathological images to retain critical color-based features like nuclear chromasia, eosinophilic cytoplasm, and staining gradients. |
| **Lack of Model Interpretability** | **Explainable AI Integration**<br>Incorporate Grad-CAM and Grad-CAM++ to highlight discriminative regions, improving transparency and clinical trust in model decisions. |

### Proposed Solution: Code-Based Implementation
Each of the above ideas has been implemented and evaluated through modular configurations in the Moonlight framework. Key implementation details include:

- **Architectures Used:**  
  A diverse set of models were evaluated, including:
  - `DenseNet121` (pretrained on ImageNet)
  - `ResidualModel` (custom ResNet-like with skip connections)
  - `EfficientConvModel` (using depthwise separable convolutions)
  - `UNetClassifier` (U-Net encoder-style downsampling with residual blocks)
  - `SimpleCNN` (lightweight 3-layer convolutional network)

- **Loss Functions Supported:**  
  - `cross_entropy`: standard log loss
  - `focal`: with tunable γ and α for hard example mining
  - `composite`: α·CrossEntropy + β·Perceptual (hybrid)

- **Image Input Mode:**  
  - Images are processed in **RGB mode** (not grayscale), retaining color histopathology features essential for diagnosis.

- **Augmentation Strategies:**
  - **Basic Augmentation**:
    - `Resize(224×224)`: standardizes input dimensions
    - `Normalize(mean=0.5, std=0.5)`: applied to each RGB channel
    - Applied to all training images with no variability

  - **Advanced Augmentation** *(used to simulate histological slide variation)*:
    - `RandomResizedCrop(224)`: simulates zoom/scaling variability
    - `RandomHorizontalFlip(p=0.5)`: mirrors tissue orientation
    - `RandomRotation(degrees=15)`: reflects natural slide rotation
    - `ColorJitter`: modifies brightness, contrast, and saturation to emulate staining variations

- **Training Setup:**
  - Epochs: 15  
  - Optimizer: Adam with AMSGrad  
  - Scheduler: StepLR (γ=0.1)
  - Batch Size: 32  
  - Stratified validation split: 10%  
  - Test split: 15% (held-out)

- **Interpretability & XAI:**
  - Manual implementations of `Grad-CAM` and `Grad-CAM++`  
  - Heatmaps generated for both benign and malignant predictions  
  - Visual overlays highlight discriminative tissue regions based on model attention
 
- **Modular Configuration & CLI Customization:**
  The Moonlight framework is fully modular and designed for reproducibility and flexibility. All training and testing parameters can be controlled via:
  - A central `config.json` file
  - Or overridden dynamically through the command line using flags

  #### 📦 Configurable Components

  | Component        | CLI Flag              | Available Options                                  |
  |------------------|------------------------|----------------------------------------------------|
  | Model Architecture | `--model_arch`       | `densenet121`, `residual`, `efficient`, `unet`, `simplecnn` |
  | Loss Function      | `--loss_fn`          | `cross_entropy`, `focal`, `composite`               |
  | Augmentation       | `--augment`          | `basic`, `advanced`                                |
  | Batch Size         | `--bs`               | e.g., `16`, `32`, `64`                             |
  | Learning Rate      | `--lr`               | e.g., `0.001`, `0.0001`                            |

  #### 🧪 Example: Run a custom training experiment
  ```bash
  python train.py -c config.json --model_arch unet --loss_fn composite --augment advanced  --bs 32 --lr 0.0005

These innovations collectively aim to improve prediction accuracy, interpretability, and trust in deep learning-based breast cancer diagnosis tools.

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
