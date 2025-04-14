**Skin Disease Classification Using Transfer Learning**
This project focuses on the classification of skin diseases using a deep learning model trained on the HAM10000 and hmnist_28_28_RGB datasets. It utilizes data augmentation, class balancing techniques, and a custom CNN architecture combining Inception, VGG, and SqueezeNet blocks.

📁 Dataset
HAM10000_metadata.csv: Metadata including diagnosis, age, sex, and localization.

hmnist_28_28_RGB.csv: RGB image data in flattened format (10015 samples, 28x28x3).

newdata.csv: Balanced version of the dataset using SMOTE + undersampling.

🔍 Workflow
Exploratory Data Analysis

Frequency of skin disease types

Demographics: age, sex, and affected body parts

Data Preprocessing

One-hot encoding, normalization

Class imbalance handled using SMOTE and Random UnderSampling

Model Building

Custom CNN using:

Inception-like block

VGG-like layers

SqueezeNet features

Evaluation

Accuracy, Confusion Matrix, ROC-AUC

KNN for similarity retrieval

🧠 Model Summary
Input: (28, 28, 3)

Final Layers: Dense (256 → 64 → 7) with softmax

Optimizer: Adam

Activation: ReLU

📊 Results
Balanced data across all 7 classes

Achieved training accuracy around 74%

Visualization: Class distribution, gender distribution, body part distribution

🛠️ Libraries Used
TensorFlow, Keras

Scikit-learn, imbalanced-learn

Seaborn, Matplotlib, Plotly
