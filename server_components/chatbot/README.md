# 🚀 차집사 Intent Classification System

A BERT-based intent classification system for promotional call centers, specifically designed for car insurance dealer outreach.

## 📋 Overview

This system classifies dealer responses during promotional calls into 8 intents:
- `fee_question` - 수수료/혜택 문의
- `about_company` - 회사 정보 확인  
- `more_questions` - 서비스/절차 문의
- `positive` - 긍정 응답
- `rejection` - 거절/보류
- `other` - 기타 (다른 거래처)
- `fallback` - 추가 정보 요청
- `greeting` - 인사

## 🏗️ Project Structure

```
.
├── data/
│   ├── intent_dataset.csv              # Original dataset (cleaned)
│   ├── intent_dataset_enhanced.csv     # With synthetic data
│   └── promotion_call.csv              # Agent call scripts
├── model/
│   ├── best_model.pth                  # Best trained model
│   ├── final_model.pth                 # Final checkpoint
│   ├── confusion_matrix.png            # Evaluation metrics
│   └── conversation_flow_guide.txt     # Usage guide
├── train.py                            # Complete training pipeline
├── inference.py                        # Inference & response system
├── generate_synthetic_data.py          # Data augmentation
└── README.md                           # This file
```

## 🔧 Requirements

```bash
pip install torch transformers pandas numpy scikit-learn tqdm matplotlib seaborn
```

**Recommended:**
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM

## 🚀 Quick Start

### Step 1: Generate Enhanced Dataset (Optional but Recommended)

```bash
python generate_synthetic_data.py
```

This will:
- Generate 25 synthetic samples per class
- Merge with existing data
- Create `intent_dataset_enhanced.csv`
- Output conversation flow guide

**Expected output:**
```
📊 Existing dataset: ~450 samples
📊 Synthetic dataset: ~200 samples
✅ Combined dataset: ~650 samples
```

### Step 2: Train the Model

```bash
python train.py
```

**Training Configuration:**
- Model: `klue/bert-base`
- Epochs: 8 (with early stopping)
- Batch Size: 8
- Learning Rate: 3e-5
- Data Augmentation: Enabled by default

**Expected Training Time:**
- CPU: ~30-45 minutes
- GPU: ~5-10 minutes

**What happens during training:**
1. ✅ Loads and augments data (2-3x original size)
2. ✅ Stratified train/val split (85/15)
3. ✅ Class-weighted sampling for imbalanced data
4. ✅ Training with gradient clipping
5. ✅ Early stopping (patience=3)
6. ✅ Saves best model based on validation accuracy
7. ✅ Generates confusion matrix
8. ✅ Prints classification report

### Step 3: Test the Model

```bash
python inference.py
```

This will:
- Load the best trained model
- Run sample predictions
- Show agent response recommendations

**Example Output:**
```
Dealer: 몇 퍼센트 주시는 거예요?
Intent: fee_question (98.5%)
Agent: 보험료의 7%를 소개료로 익일 지급해드립니다.
```

## 📊 Model Performance

**Expected Results (with augmentation):**

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95-98% |
| Validation Accuracy | ~85-92% |
| Best for Small Dataset | ✅ |

**Class-wise Performance:**
- High accuracy: `fee_question`, `greeting`, `positive`
- Medium accuracy: `about_company`, `more_questions`, `fallback`
- May need more data: `other`, `rejection` (very similar patterns)

## 🎯 Usage in Production

### Basic Inference

```python
from inference import IntentClassifier

# Load model
classifier = IntentClassifier('./model/best_model.pth')

# Single prediction
dealer_text = "몇 프로 주시는 건가요?"
intent, confidence = classifier.predict(dealer_text)
print(f"Intent: {intent} ({confidence:.2%})")

# Get suggested agent response
response = classifier.get_response(dealer_text)
print(f"Agent should say: {response['agent_response']}")
```

### Batch Processing

```python
# Process multiple responses
dealer_responses = [
    "차집사가 어디예요?",
    "괜찮네요 명함 주세요",
    "저 다른 데 하고 있어요"
]

results = classifier.predict_batch(dealer_responses)
print(results)
```

### Interactive Testing

```python
classifier = IntentClassifier('./model/best_model.pth')
classifier.interactive_test()
```

## 🎓 Training Parameters Explained

### Core Settings

```python
# In train.py
USE_AUGMENTATION = True      # Enable data augmentation
AUGMENT_MULTIPLIER = 3       # 3x augmentation (recommended)
EPOCHS = 8                   # Max epochs (early stopping active)
BATCH_SIZE = 8               # Small batch for better gradients
LEARNING_RATE = 3e-5         # Optimal for BERT fine-tuning
MAX_LEN = 64                 # Sufficient for short dealer responses
```

### When to Adjust

**If you have more data (>1000 samples):**
```python
USE_AUGMENTATION = False     # Disable if data is sufficient
BATCH_SIZE = 16              # Larger batch
EPOCHS = 5                   # Fewer epochs needed
```

**If validation accuracy is low (<80%):**
```python
AUGMENT_MULTIPLIER = 5       # More augmentation
EPOCHS = 10                  # More training
LEARNING_RATE = 2e-5         # Lower learning rate
```

**If overfitting (train acc >> val acc):**
```python
AUGMENT_MULTIPLIER = 5       # More diverse data
# Add dropout in model (requires code change)
```

## 📈 Data Augmentation Strategies

The system uses 3 augmentation techniques:

### 1. Synonym Replacement
```
Original: "몇 프로에요?"
Augmented: "얼마 퍼센트에요?"
```

### 2. Ending Variation
```
Original: "괜찮아요"
Augmented: "괜찮네요", "괜찮군요", "괜찮죠"
```

### 3. Combined
```
Original: "몇 프로 주는데요?"
Augmented: "얼마 퍼센트 주나요?"
```

## 🔄 Conversation Flow Integration

The model works best as part of a conversation flow system:

```
1. Agent starts call (uses promotion_call.csv scripts)
2. Dealer responds
3. Model classifies intent
4. System selects appropriate response
5. Agent continues based on intent

Example Flow:
Agent: "보험료의 7%를 지급해드립니다"
Dealer: "몇 프로요?" [Detected: fee_question]
Agent: "7프로이며 익일 지급됩니다" [Clarification]
Dealer: "오 괜찮네요" [Detected: positive]
Agent: "명함 보내드릴게요" [Close with materials]
```

See `model/conversation_flow_guide.txt` for detailed flow logic.

## 🐛 Troubleshooting

### Issue: Low Validation Accuracy (<70%)

**Solution:**
1. Generate more synthetic data: `AUGMENT_MULTIPLIER = 5`
2. Increase training epochs: `EPOCHS = 12`
3. Check data quality - remove duplicates/mislabeled samples

### Issue: Model predicts same class for everything

**Solution:**
1. Check class imbalance in data
2. Ensure `WeightedRandomSampler` is working
3. Verify label mapping is correct

### Issue: "CUDA out of memory"

**Solution:**
```python
BATCH_SIZE = 4  # Reduce batch size
MAX_LEN = 32    # Reduce sequence length
```

### Issue: Training is too slow (CPU)

**Solution:**
- Use Google Colab (free GPU)
- Reduce augmentation: `AUGMENT_MULTIPLIER = 2`
- Or wait patiently (~30-45 min on CPU)

## 📝 Customization

### Adding New Intent Classes

1. **Update dataset:**
```csv
question,label
새로운 질문,new_intent_class
```

2. **Update label mapping in train.py:**
```python
label2id = {
    # ... existing labels ...
    "new_intent_class": 8
}
NUM_LABELS = 9  # Update count
```

3. **Add response template in inference.py:**
```python
self.response_templates = {
    # ... existing templates ...
    "new_intent_class": [
        "새로운 응답 템플릿"
    ]
}
```

4. **Retrain model**

### Using Different Base Model

```python
# In train.py
MODEL_NAME = "klue/roberta-base"  # Instead of klue/bert-base
# or
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
```

## 📊 Model Checkpoints

The training saves two checkpoints:

- **`best_model.pth`** - Best validation accuracy (use this)
- **`final_model.pth`** - Final epoch (for comparison)

Each checkpoint contains:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'val_acc': float,
    'val_loss': float,
    'label2id': dict,
    'id2label': dict,
    'tokenizer_name': str,
    'max_len': int
}
```

## 🔐 Best Practices

### Data Quality
- ✅ Remove duplicates before training
- ✅ Fix typos in your dataset
- ✅ Ensure consistent labeling
- ✅ Balance class distribution (use augmentation)

### Training
- ✅ Always use validation set
- ✅ Monitor overfitting (train vs val accuracy)
- ✅ Use early stopping
- ✅ Save best model, not last

### Deployment
- ✅ Test on real dealer responses
- ✅ Log predictions for analysis
- ✅ Retrain periodically with new data
- ✅ Have fallback for low-confidence predictions

## 📞 Integration Example

```python
# Simplified call center integration
class CallCenterAssistant:
    def __init__(self):
        self.classifier = IntentClassifier('./model/best_model.pth')
        self.conversation_history = []
    
    def process_dealer_response(self, dealer_text):
        # Classify intent
        result = self.classifier.get_response(dealer_text)
        
        # Log conversation
        self.conversation_history.append({
            'dealer': dealer_text,
            'intent': result['detected_intent'],
            'confidence': result['confidence'],
            'agent_response': result['agent_response']
        })
        
        # Return suggested response
        return result['agent_response']
    
    def should_end_call(self):
        # End if 2+ consecutive rejections
        if len(self.conversation_history) >= 2:
            last_two = self.conversation_history[-2:]
            if all(h['intent'] == 'rejection' for h in last_two):
                return True
        return False
```

## 📚 Additional Resources

- [KLUE BERT Documentation](https://github.com/KLUE-benchmark/KLUE)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Korean NLP Resources](https://github.com/songys/AwesomeKorean_Data)

## 🤝 Contributing

To improve the model:

1. Add more training samples (especially for low-accuracy classes)
2. Try different augmentation techniques
3. Experiment with model architectures
4. Share your results!

## 📄 License

This project is for internal use in promotional call centers.

## ⚠️ Important Notes

- Model accuracy depends heavily on training data quality
- Regular retraining recommended as new patterns emerge
- Always have human oversight for critical decisions
- Low confidence predictions (<70%) should be escalated to supervisors

---

**Questions or Issues?**

Check the conversation flow guide in `model/conversation_flow_guide.txt` for usage patterns!