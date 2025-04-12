def get_model(config):
    model_type = config['arch']['type']

    if model_type == 'simple_cnn':
        from .simple_cnn import SimpleCNN
        return SimpleCNN(**config['arch'].get('args', {}))

    elif model_type == 'residual':
        from .residual_model import ResidualModel
        return ResidualModel(**config['arch'].get('args', {}))

    elif model_type == 'efficient':
        from .efficient_model import EfficientConvModel
        return EfficientConvModel(**config['arch'].get('args', {}))

    elif model_type == 'densenet121':
        from .model import densenet121
        return densenet121(**config['arch'].get('args', {}))

    elif model_type == 'unet_classifier':
        from .unet_classifier import UNetClassifier
        return UNetClassifier(**config['arch'].get('args', {}))

    else:
        raise ValueError(f"Unknown model type: {model_type}")
