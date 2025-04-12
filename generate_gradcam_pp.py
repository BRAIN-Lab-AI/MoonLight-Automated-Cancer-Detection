import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from parse_config import ConfigParser
from model import get_model

# ==== CONFIG ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed image paths
benign_path = "data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549CD/400X/SOB_B_A-14-22549CD-400-027.png"
malig_path  = "data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-18842D/400X/SOB_M_MC-14-18842D-400-014.png"

# Image preprocessing
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
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()

    A = activations[0].detach()
    dY = gradients[0].detach()

    weights = dY.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * A).sum(dim=1)).squeeze().cpu().numpy()

    cam -= cam.min()
    cam /= cam.max()
    cam = cv2.resize(cam, (224, 224))

    handle_fw.remove()
    handle_bw.remove()
    return cam

def gradcam_plus(model, img_tensor, target_layer):
    model.eval()
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
    class_score = output[0, pred_class]
    class_score.backward()

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
    cam -= cam.min()
    cam /= cam.max()
    cam = cv2.resize(cam, (224, 224))

    handle_fw.remove()
    handle_bw.remove()
    return cam, pred_class, confidence, probs



def save_overlay(orig_img, cam_plain, cam_pp, output_path, pred_label, confidence):
    orig_resized = orig_img.resize((224, 224))
    orig_np = np.array(orig_resized).astype(np.uint8)

    # Convert CAMs to heatmap overlays
    def apply_heatmap(cam):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        return cv2.addWeighted(heatmap, 0.4, orig_np, 0.6, 0)

    overlay_grad = apply_heatmap(cam_plain)
    overlay_gradpp = apply_heatmap(cam_pp)


    # Add spacing between images (e.g., 10px white line)
    gap = 10
    spacer = np.ones((224, gap, 3), dtype=np.uint8) * 255  # vertical white bar

    # Combine side by side with spacing
    combined = np.hstack((orig_np, spacer, overlay_grad, spacer, overlay_gradpp))

    # Add prediction text bar at the top
    bar_height = 40
    label_bar = np.ones((bar_height, combined.shape[1], 3), dtype=np.uint8) * 255
    label_text = f"Prediction: {pred_label} ({confidence:.2f})"
    cv2.putText(label_bar, label_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    final_image = np.vstack((label_bar, combined))
    cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

def run_gradcam_pp(config_path, resume_path):
    # Load JSON config file and build ConfigParser
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = ConfigParser(config_dict)
    config.resume = resume_path
    config.device = None

    model = get_model(config).to(device)
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])


    # === Auto-select target layer ===
    target_layer = None

    if hasattr(model, "features"):  # DenseNet or ResNet
        try:
            target_layer = model.features.denseblock3.denselayer16.conv2
        except AttributeError:
            pass

    # Fallback: find the last 2D convolution layer in the model
    if target_layer is None:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                print(f"✅ Auto-selected target layer: {module}")
                break

    if target_layer is None:
        raise RuntimeError("❌ Could not determine target_layer for Grad-CAM++. Please update manually.")

    class_names = ['benign', 'malignant']

    for path, label in [(benign_path, "benign"), (malig_path, "malignant")]:
        orig_img, img_tensor = load_image(path)
        # Run Grad-CAM++ and get prediction
        cam_pp, pred_class, confidence, probs = gradcam_plus(model, img_tensor, target_layer)
        pred_label = class_names[pred_class]

        # Also compute Grad-CAM (vanilla)
        cam_plain = gradcam(model, img_tensor, target_layer, pred_class)

        print(f"[{label}] Prediction: {pred_label} ({confidence:.2f})")
        print(f"Probabilities → benign: {probs[0]:.2f} | malignant: {probs[1]:.2f}")
        exp_dir = os.path.dirname(resume_path)
        exp_name = os.path.basename(exp_dir)
        output_path = os.path.join(exp_dir, f"{exp_name}_cam_{label}.png")
        save_overlay(orig_img, cam_plain, cam_pp, output_path, pred_label, confidence)
        print(f"Saved: cam_{label}.png\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    parser.add_argument('-r', '--resume', required=True, type=str, help='Path to checkpoint')
    args = parser.parse_args()

    run_gradcam_pp(args.config, args.resume)
