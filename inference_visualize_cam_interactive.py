import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from parse_config import ConfigParser
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img, img_tensor

def gradcam(model, img_tensor, target_layer, pred_class):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()

    A = activations[0].detach()
    dY = gradients[0].detach()
    weights = dY.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * A).sum(dim=1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (224, 224))

    handle_fw.remove()
    handle_bw.remove()
    return cam

def gradcam_plus(model, img_tensor, target_layer):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    probs = F.softmax(output, dim=1).squeeze().detach().cpu().numpy()
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]

    model.zero_grad()
    score = output[0, pred_class]
    score.backward()

    A = activations[0].detach()
    dY = gradients[0].detach()
    d2Y = dY ** 2
    d3Y = dY ** 3

    eps = 1e-8
    alpha_num = d2Y
    alpha_denom = 2 * d2Y + A * d3Y.sum(dim=(2, 3), keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)
    alpha = alpha_num / alpha_denom
    weights = (alpha * F.relu(dY)).sum(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * A).sum(dim=1)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (224, 224))

    handle_fw.remove()
    handle_bw.remove()
    return cam, pred_class, confidence, probs

def visualize_cam(img_path, config_path, checkpoint_path):
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = ConfigParser(config_dict)
    config.resume = checkpoint_path
    config.device = None

    model = get_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Auto-detect target layer
    target_layer = None
    if hasattr(model, "features"):
        try:
            target_layer = model.features.denseblock3.denselayer16.conv2
        except AttributeError:
            pass
    if target_layer is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    orig_img, img_tensor = load_image(img_path)
    cam_pp, pred_class, confidence, probs = gradcam_plus(model, img_tensor, target_layer)
    cam_plain = gradcam(model, img_tensor, target_layer, pred_class)

    # Overlay heatmaps
    def overlay(cam):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        orig_np = np.array(orig_img.resize((224, 224))).astype(np.uint8)
        return cv2.addWeighted(heatmap, 0.4, orig_np, 0.6, 0)

    overlay_plain = overlay(cam_plain)
    overlay_pp = overlay(cam_pp)

    # Display with matplotlib
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(orig_img.resize((224, 224)))
    axs[0].set_title("Original")
    axs[1].imshow(overlay_plain)
    axs[1].set_title("Grad-CAM")
    axs[2].imshow(overlay_pp)
    axs[2].set_title("Grad-CAM++")
    for ax in axs:
        ax.axis('off')
    fig.suptitle(f"Prediction: {'benign' if pred_class == 0 else 'malignant'} ({confidence:.2f})", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, type=str, help="Path to input image")
    parser.add_argument('-c', '--config', required=True, type=str, help="Path to config file")
    parser.add_argument('-r', '--resume', required=True, type=str, help="Path to model checkpoint")
    args = parser.parse_args()
    visualize_cam(args.image, args.config, args.resume)
