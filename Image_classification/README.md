# Image Classifier - TensorFlow & Streamlit

A modern image classification application built with TensorFlow and Streamlit. The model is trained on the CIFAR-10 dataset to classify images into 10 different categories.

## Features

- **Real-time Image Classification** - Upload any image and get instant predictions
- **Confidence Scores** - View detailed confidence percentages for each class
- **Sample Predictions** - Test the model with sample images from the dataset
- **Beautiful UI** - Clean, intuitive Streamlit interface
- **High Accuracy** - CNN model trained on CIFAR-10 dataset

## Categories

The model can classify images into these 10 categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Installation

1. Clone this repository:
\`\`\`bash
git clone https://github.com/aayushsingh047/Image_classification
cd image-classifier
\`\`\`

2. Create a virtual environment (recommended):
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

### 1. Train the Model

First, train the image classifier model:
\`\`\`bash
python train_model.py
\`\`\`

This will:
- Download the CIFAR-10 dataset (automatic)
- Train a CNN model for 20 epochs
- Save the trained model to `models/cifar10_classifier.h5`
- Save class names to `models/class_names.json`

### 2. Run the Streamlit App

\`\`\`bash
streamlit run app.py
\`\`\`

The app will open in your browser at `http://localhost:8501`

## Project Structure

\`\`\`
image-classifier/
├── train_model.py           # Model training script
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Trained model files (created after training)
│   ├── cifar10_classifier.h5
│   └── class_names.json
└── .gitignore
\`\`\`

## How It Works

1. **Training Phase** (`train_model.py`):
   - Loads CIFAR-10 dataset (50,000 training + 10,000 test images)
   - Builds a Convolutional Neural Network (CNN)
   - Trains for 20 epochs with 128 batch size
   - Achieves ~75% accuracy on test set

2. **Prediction Phase** (`app.py`):
   - Loads the trained model
   - Preprocesses uploaded images (resize to 32x32, normalize)
   - Makes predictions and returns confidence scores
   - Displays results with confidence visualization

## Model Architecture

\`\`\`
Input (32x32x3)
    ↓
Conv2D(32 filters, 3x3) + ReLU
    ↓
MaxPooling2D(2x2)
    ↓
Conv2D(64 filters, 3x3) + ReLU
    ↓
MaxPooling2D(2x2)
    ↓
Conv2D(64 filters, 3x3) + ReLU
    ↓
Flatten
    ↓
Dense(64) + ReLU + Dropout(0.5)
    ↓
Dense(10) + Softmax
    ↓
Output (10 classes)
\`\`\`

## Performance

- **Training Accuracy**: ~80%
- **Test Accuracy**: ~75%
- **Training Time**: ~5-10 minutes (on CPU)
- **Inference Time**: <100ms per image

## Technologies Used

- **TensorFlow/Keras** - Deep Learning Framework
- **Streamlit** - Web Application Framework
- **NumPy** - Numerical Computing
- **Pillow** - Image Processing
- **CIFAR-10** - Dataset

## Tips for Better Predictions

1. Upload clear, well-lit images
2. The model works best with objects similar to CIFAR-10 dataset
3. Center the object in the image
4. Use standard image formats (JPG, PNG)

## Deployment Options

### Streamlit Cloud (Free)
\`\`\`bash
streamlit run app.py --logger.level=error
\`\`\`

Then deploy directly to [Streamlit Cloud](https://streamlit.io/cloud)

### Docker
\`\`\`dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
\`\`\`


## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Created with ❤️ using TensorFlow and Streamlit**
