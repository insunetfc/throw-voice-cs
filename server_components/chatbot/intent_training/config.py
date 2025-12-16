"""
Configuration file for intent classification training.
Modify these parameters to tune your model.
"""

import torch

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_CONFIG = {
    # Base model to use (Korean BERT models)
    'model_name': 'klue/bert-base',
    # Alternatives:
    # 'klue/roberta-base'
    # 'monologg/koelectra-base-v3-discriminator'
    # 'klue/bert-base' (recommended for Korean)
    
    # Number of intent classes
    'num_labels': 8,
    
    # Maximum sequence length (characters)
    # Most dealer responses are short, so 64 is usually enough
    'max_length': 64,
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
TRAINING_CONFIG = {
    # Number of training epochs
    'epochs': 8,
    
    # Batch size (reduce if OOM errors)
    'batch_size': 8,
    
    # Learning rate
    'learning_rate': 3e-5,
    
    # Weight decay for regularization
    'weight_decay': 0.01,
    
    # Warmup proportion (10% of total steps)
    'warmup_proportion': 0.1,
    
    # Gradient clipping threshold
    'max_grad_norm': 1.0,
    
    # Random seed for reproducibility
    'seed': 42,
    
    # Early stopping patience (epochs)
    'patience': 3,
    
    # Validation split ratio
    'val_split': 0.15,
}

# ============================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================
AUGMENTATION_CONFIG = {
    # Enable/disable data augmentation
    'enabled': True,
    
    # How many augmented samples per original
    # Recommended: 2-3 for small datasets (<500 samples)
    #              1-2 for medium datasets (500-1000)
    #              0 for large datasets (>1000)
    'multiplier': 3,
    
    # Balance classes by oversampling minority classes
    'balance_classes': True,
    
    # Use weighted sampling during training
    'use_weighted_sampler': True,
}

# ============================================================
# PATH CONFIGURATION
# ============================================================
PATH_CONFIG = {
    # Data directory
    'data_dir': './data',
    
    # Original dataset filename
    'dataset_file': 'intent_dataset.csv',
    
    # Enhanced dataset filename (with synthetic data)
    'enhanced_dataset_file': 'intent_dataset_enhanced.csv',
    
    # Model save directory
    'model_dir': './model',
    
    # Best model filename
    'best_model_file': 'best_model.pth',
    
    # Final model filename
    'final_model_file': 'final_model.pth',
    
    # Confusion matrix save path
    'confusion_matrix_file': 'confusion_matrix.png',
}

# ============================================================
# LABEL CONFIGURATION
# ============================================================
LABEL_CONFIG = {
    # Label to ID mapping
    'label2id': {
        "fee_question": 0,      # 수수료/혜택 문의
        "about_company": 1,     # 회사 정보 확인
        "more_questions": 2,    # 서비스/절차 문의
        "positive": 3,          # 긍정 응답
        "rejection": 4,         # 거절/보류
        "other": 5,             # 기타(다른 거래처)
        "fallback": 6,          # 추가 정보 요청
        "greeting": 7           # 인사
    },
    
    # Korean label names for display
    'label_names_kr': {
        "fee_question": "수수료 문의",
        "about_company": "회사 정보",
        "more_questions": "서비스 문의",
        "positive": "긍정 응답",
        "rejection": "거절",
        "other": "기존 거래처",
        "fallback": "자료 요청",
        "greeting": "인사"
    }
}

# Derive id2label automatically
LABEL_CONFIG['id2label'] = {v: k for k, v in LABEL_CONFIG['label2id'].items()}

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE_CONFIG = {
    # Automatically use CUDA if available
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # Force CPU (useful for debugging)
    'force_cpu': False,
}

# Override device if force_cpu is True
if DEVICE_CONFIG['force_cpu']:
    DEVICE_CONFIG['device'] = torch.device('cpu')

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOGGING_CONFIG = {
    # Print training progress every N batches
    'log_interval': 10,
    
    # Show progress bars
    'show_progress': True,
    
    # Verbose mode (more detailed output)
    'verbose': True,
}

# ============================================================
# INFERENCE CONFIGURATION
# ============================================================
INFERENCE_CONFIG = {
    # Confidence threshold for predictions
    # Predictions below this will be marked as uncertain
    'confidence_threshold': 0.7,
    
    # Return top-k predictions
    'top_k': 3,
    
    # Use ensemble (not implemented yet)
    'use_ensemble': False,
}

# ============================================================
# PRESETS FOR DIFFERENT SCENARIOS
# ============================================================

def get_preset(preset_name):
    """Get predefined configuration presets"""
    
    presets = {
        'small_dataset': {
            # For datasets < 500 samples
            'TRAINING_CONFIG': {
                'epochs': 10,
                'batch_size': 8,
                'learning_rate': 3e-5,
            },
            'AUGMENTATION_CONFIG': {
                'enabled': True,
                'multiplier': 5,
                'balance_classes': True,
            }
        },
        
        'medium_dataset': {
            # For datasets 500-1000 samples
            'TRAINING_CONFIG': {
                'epochs': 6,
                'batch_size': 16,
                'learning_rate': 2e-5,
            },
            'AUGMENTATION_CONFIG': {
                'enabled': True,
                'multiplier': 2,
                'balance_classes': True,
            }
        },
        
        'large_dataset': {
            # For datasets > 1000 samples
            'TRAINING_CONFIG': {
                'epochs': 4,
                'batch_size': 32,
                'learning_rate': 2e-5,
            },
            'AUGMENTATION_CONFIG': {
                'enabled': False,
                'multiplier': 0,
                'balance_classes': False,
            }
        },
        
        'quick_test': {
            # For quick testing/debugging
            'TRAINING_CONFIG': {
                'epochs': 2,
                'batch_size': 4,
                'learning_rate': 5e-5,
            },
            'AUGMENTATION_CONFIG': {
                'enabled': False,
                'multiplier': 0,
            }
        },
        
        'production': {
            # Optimized for production deployment
            'TRAINING_CONFIG': {
                'epochs': 8,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'patience': 5,
            },
            'AUGMENTATION_CONFIG': {
                'enabled': True,
                'multiplier': 3,
                'balance_classes': True,
            }
        }
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    return presets[preset_name]


def apply_preset(preset_name):
    """Apply a preset configuration"""
    preset = get_preset(preset_name)
    
    # Update global configs
    for config_name, config_values in preset.items():
        globals()[config_name].update(config_values)
    
    print(f"✅ Applied preset: {preset_name}")


# ============================================================
# EXPORT ALL CONFIGS
# ============================================================
def get_all_configs():
    """Get all configurations as a single dictionary"""
    return {
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'paths': PATH_CONFIG,
        'labels': LABEL_CONFIG,
        'device': DEVICE_CONFIG,
        'logging': LOGGING_CONFIG,
        'inference': INFERENCE_CONFIG,
    }


def print_config():
    """Print current configuration"""
    configs = get_all_configs()
    
    print("\n" + "="*60)
    print("📋 CURRENT CONFIGURATION")
    print("="*60)
    
    for section_name, section_config in configs.items():
        print(f"\n[{section_name.upper()}]")
        for key, value in section_config.items():
            print(f"  {key:25s} = {value}")
    
    print("\n" + "="*60 + "\n")


# ============================================================
# USAGE EXAMPLES
# ============================================================
if __name__ == "__main__":
    print("\n🔧 Configuration Examples\n")
    
    print("1. Print current configuration:")
    print("   from config import print_config")
    print("   print_config()")
    
    print("\n2. Apply a preset:")
    print("   from config import apply_preset")
    print("   apply_preset('small_dataset')")
    
    print("\n3. Use in training script:")
    print("   from config import MODEL_CONFIG, TRAINING_CONFIG")
    print("   model_name = MODEL_CONFIG['model_name']")
    print("   epochs = TRAINING_CONFIG['epochs']")
    
    print("\n4. Available presets:")
    presets = ['small_dataset', 'medium_dataset', 'large_dataset', 'quick_test', 'production']
    for preset in presets:
        print(f"   - {preset}")
    
    print("\n" + "="*60)
    print("Current Configuration:")
    print("="*60)
    print_config()
