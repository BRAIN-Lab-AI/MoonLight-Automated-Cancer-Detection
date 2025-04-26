import os
from collections import defaultdict

# Dataset Root Path
DATASET_DIR = 'data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'

# Initialize counters
counts = {
    'benign': defaultdict(int),
    'malignant': defaultdict(int)
}

# Loop through benign and malignant
for cls in ['benign', 'malignant']:
    cls_path = os.path.join(DATASET_DIR, cls)

    for root, dirs, files in os.walk(cls_path):
        for dir_name in dirs:
            if dir_name in ['40X', '100X', '200X', '400X']:
                mag_folder = os.path.join(root, dir_name)
                num_images = len([
                    file for file in os.listdir(mag_folder)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                counts[cls][dir_name] += num_images

# Print results
print("\nTotal image counts per class and magnification:\n")
print(f"{'Class':<10} {'40X':>6} {'100X':>6} {'200X':>6} {'400X':>6}")
print("-" * 35)
for cls in ['benign', 'malignant']:
    line = f"{cls:<10}"
    for mag in ['40X', '100X', '200X', '400X']:
        line += f" {counts[cls][mag]:6}"
    print(line)

# Optionally print Grand Total
print("\nGrand totals:")
for mag in ['40X', '100X', '200X', '400X']:
    total = counts['benign'][mag] + counts['malignant'][mag]
    print(f"{mag}: {total} images")
