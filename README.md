# Recyclable Waste Classifier

This is a computer vision project that classifies images of waste as recyclable or non-recyclable.
It uses a Convolutional Neural Network (CNN) trained on a real-world dataset collected by two Stanford students, repository linked here: https://github.com/garythung/trashnet.


## Model overview
- **Architecture**: 2-layer CNN with ReLU activations and MaxPooling
- **Input size**: 2,527 RGB images, dimensions 64 x 64
- **Sampling**: WeightedRandomSampler for handling severe class imbalance (95-5)
- **Loss function**: BCEWithLogitsLoss used with class imbalance correction
- **Data augmentation**: Training data augmented with rotation and horizontal flip


## Results

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| Non-Recyclable   | 0.44      | 0.68   | 0.54     | 28      |
| Recyclable       | 0.98      | 0.95   | 0.96     | 478     |
| **Accuracy**     |           |        | **0.93** | 506     |

- Strong recall on the minority class, making the model practical for deployment (high FNR is more costly).
- Model was trained for 10 epochs using Adam optimizer with learning rate of `0.001`.
  

## File Structure
- train.py - Main training script
- waste_classifier.pth - Saved model weights
- data/dataset-resized/ - Folder of images organized by class
- README.md


## How to Run
```bash
pip install torch torchvision matplotlib scikit-learn
python train.py


## Future Improvements
- Using a pre-trained model like ResNet-18 for better feature extraction
- Training the model on a larger dataset from unique consumer environments to specialize the model for commercial use
- Developing a web interface for real-time predictions (e.g., Streamlit, Flask)
