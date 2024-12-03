# Predicting Modified Overall Disability Index (MODI) Pain Scores from Full-Body X-rays

This project involves prediction of  the Modified Overall Disability Index (MODI) pain score using full-body X-ray images. The pipeline is designed to overcome the challenges posed by a limited dataset by leveraging advanced machine learning techniques.

## Objective

- **Score Prediction:** Predict a pain score (MODI) for each patient based on two X-ray views: **front** and **side**.
- **Heatmap Generation:** Localize regions in the X-rays contributing significantly to the pain score.

## Challenges

- **Limited Dataset:** Only 150 image pairs are available.
- **Dual Input:** Incorporating two distinct views (front and side) for a single prediction.

## Approach

### Data Preprocessing

- Two images (front and side views) are available for each patient.
- Images are preprocessed for feature extraction, maintaining compatibility with pre-trained models.

### Model Architecture

1. **Feature Extraction:**  
   - A dual-input architecture processes both front and side views independently.  
   - Various pre-trained models are evaluated for feature extraction, including:
     - **ResNet50**
     - **Vision Transformers (ViTs)**
     - Models fine-tuned on medical datasets (e.g., chest X-rays).

2. **Fusion Mechanisms:**  
   - Features extracted from both views are combined using:
     - **Concatenation:** A simple yet effective strategy to merge feature vectors.
     - **Cross-Attention Mechanisms:** Enhances interaction between features from the two views for better representation.

3. **Regression Head:**  
   - A regression layer predicts the pain score based on the fused feature representation.

### Transfer Learning

Given the limited dataset, transfer learning is employed to leverage knowledge from pre-trained models. Fine-tuning on the available data ensures the model adapts to the specific domain.

## Heatmap Generation

- Heatmaps are generated to identify regions in the X-rays that contribute most significantly to the predicted pain score.  
- These visualizations aid in understanding the model's focus and validating its predictions.

## Current Progress

- Feature extraction and fusion techniques (concatenation) have been implemented.
- Cross-attention mechanisms are under development to improve fusion quality.

## Future Directions

- Experimentation with additional pre-trained models.
- Fine-tuning hyperparameters for better generalization.
- Enhancing heatmap generation techniques for improved interpretability.

## Repository Structure

```plaintext
.
├── dataloaders.py          # Load the X-ray image dataset (front and side views)
├── model.py                # Contains model definition
├── train.py                # Main training loop
