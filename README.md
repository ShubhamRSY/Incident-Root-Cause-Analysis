# Incident Root Cause Analysis Exercise

## Description
This repository contains the implementation of the **Incident Root Cause Analysis Exercise**, which leverages machine learning to identify the root causes of incidents in IT operations (ITOps). The project includes data preprocessing, model building, evaluation, and prediction functionalities.

---

## Table of Contents
- [What We Did](#what-we-did)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Notes](#notes)

---

## What We Did
1. **Loaded and Preprocessed the Dataset**:
   - Columns like `CPU_LOAD`, `MEMORY_LEAK_LOAD`, and `DELAY` were used as features.
   - The target column `ROOT_CAUSE` was label-encoded into numerical categories.
   - Dataset was split into training and testing sets.

2. **Built and Trained a Neural Network**:
   - Constructed a sequential neural network using TensorFlow/Keras.
   - The model consisted of:
     - Two dense layers with ReLU activation.
     - One softmax output layer for multi-class classification.
   - Key training parameters:
     - Epochs: 20
     - Batch size: 64
     - Validation split: 20%
     - Optimized using `categorical_crossentropy` loss and `accuracy` metric.

3. **Evaluated the Model**:
   - Achieved a test accuracy of **84.5%** after training.

4. **Predicted Root Causes**:
   - Implemented prediction for single instances and batches.
   - Translated predictions back to original root cause labels using the label encoder.

---

## Results
### Model Performance:
- **Accuracy**: 86.7% (validation), 84.5% (test dataset).
- **Loss**: 
  - Final validation loss: 0.41
  - Test dataset loss: 0.43

### Predictions:
- **Example Prediction**:
  Input: `[1, 0, 0, 0, 1, 1, 0]`  
  Output: `'DATABASE_ISSUE'`

- **Batch Prediction Example**:
  Input:


## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Incident-Root-Cause-Analysis-Exercise.git

2. Navigate to the project directory:
   ```bash
   cd Incident-Root-Cause-Analysis-Exercise
3.Install the required Python packages:
``` bash
   pip install -r requirements.txt
```

Usage
Load and preprocess the dataset:
Convert the ROOT_CAUSE column to numerical values using a label encoder.
Split the data into training and testing sets:
```
from sklearn.model_selection import train_test_split
# Preprocessing and splitting code here...
```

Train the model:
Build a sequential neural network using TensorFlow/Keras:
```
model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_split=0.2)
```
Evaluate the model:
```
model.evaluate(X_test, Y_test)
```
Make predictions:
For single and batch predictions:
```
prediction = model.predict(input_data)
decoded_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
```
Project Structure
- code_06_XX Incident Root Cause Analysis Exercise.html: Contains a detailed implementation of the ML pipeline.
- root_cause_analysis.csv: The dataset for the exercise.
- requirements.txt: List of dependencies for the project.
- README.md: This file.
- Additional generated files or visualizations (to be added as needed).

Dependencies
Python 3.7 or higher
TensorFlow
scikit-learn
Pandas
NumPy
Matplotlib (optional, for visualizations)
Install all dependencies with:
```
pip install -r requirements.txt

```





