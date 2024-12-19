# Incident Root Cause Analysis Exercise

## Description
This repository contains a complete implementation of the **"Incident Root Cause Analysis Exercise"**, which uses machine learning models to predict the root causes of incidents in IT operations (ITOps). The project includes data preprocessing, model building, evaluation, and prediction steps.

---

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- Predicts root causes of incidents using a neural network.
- Processes structured data for root cause analysis.
- Includes batch predictions for multiple incident inputs.
- Achieves a high validation accuracy (~86%).

---

## Dataset
The dataset (`root_cause_analysis.csv`) includes the following columns:
- **ID**: Incident ID.
- **CPU_LOAD**: Indicator of CPU load.
- **MEMORY_LEAK_LOAD**: Indicator of memory leak.
- **DELAY**: Network delay indicator.
- **ERROR_1000, ERROR_1001, ERROR_1002, ERROR_1003**: Binary error flags.
- **ROOT_CAUSE**: The root cause (target variable).

---

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





