import os

class ModelConfig:
    # LSTM Configuration
    LSTM = {
        'input_size': 50,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': True,
        'output_size': 1,
    }

    # Attention Configuration
    ATTENTION = {
        'num_heads': 8,
        'hidden_size': 128,
        'dropout': 0.1,
    }

    # CNN Configuration
    CNN = {
        'num_filters': [32, 64, 128],
        'kernel_sizes': [3, 5, 7],
        'dropout': 0.2,
    }

    # Ensemble Configuration
    ENSEMBLE = {
        'models': ['lstm', 'attention', 'cnn'],
        'meta_learner_hidden': 64,
        'meta_learner_dropout': 0.1,
    }

    # Training Configuration
    TRAINING = {
        'batch_size': int(os.getenv('BATCH_SIZE', 32)),
        'epochs': int(os.getenv('EPOCHS', 100)),
        'learning_rate': float(os.getenv('LEARNING_RATE', 0.001)),
        'learning_rate_decay': 0.95,
        'early_stopping_patience': 15,
        'gradient_clip': 1.0,
        'optimizer': 'adam',
    }

    # Validation Configuration
    VALIDATION = {
        'validation_split': 0.2,
        'test_split': 0.1,
        'walk_forward_steps': 12,
        'walk_forward_overlap': 1,
    }

    # Device Configuration
    DEVICE = os.getenv('MODEL_DEVICE', 'cuda')

config = ModelConfig()
