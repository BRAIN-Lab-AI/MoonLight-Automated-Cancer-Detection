import argparse
import torch
import numpy as np
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
from model import get_model
from model.loss_functions import get_loss_function
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

def main(config, resume):
    logger = config.get_logger('test')

    # Set seed for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup test data loader
    test_loader = config.init_obj('data_loader', module_data)

    # Build and load model
    model = get_model(config)
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Get loss function
    criterion = get_loss_function(config['loss'])

    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)

            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Metrics
    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

    print(f"\n===== Test Results =====", flush=True)
    print(f"Test Loss       : {avg_loss:.4f}", flush=True)
    print(f"Test Accuracy   : {acc * 100:.2f}%", flush=True)
    print(f"Precision       : {precision:.4f}", flush=True)
    print(f"Recall          : {recall:.4f}", flush=True)
    print(f"F1 Score        : {f1:.4f}", flush=True)
    print(f"========================\n", flush=True)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Test')
    args.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args.add_argument('-r', '--resume', required=True, type=str, help='Path to model checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='Indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config, config.resume)
