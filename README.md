# Plant-Health-Detection-Aquaponics
🌱 Plant Health Classification using Neural Networks This project presents a deep learning approach to classify plant health based on input features from an agricultural dataset. The model is built using TensorFlow and Keras, and addresses real-world issues like class imbalance and noisy data using preprocessing techniques and SMOTE.

📌 Problem Statement Given a dataset with various features, the goal is to classify whether a plant is healthy or unhealthy using a feed-forward neural network.

🧠 Model Architecture A regularized neural network with:

Dense (64 units) + ReLU + L2 + BatchNormalization + Dropout

Dense (32 units) + ReLU + L2 + BatchNormalization + Dropout

Dense (16 units) + ReLU + L2 + BatchNormalization

Dense (1 unit) + Sigmoid (binary output)

⚙️ Workflow Data Preprocessing

Dropped irrelevant & null features

Encoded target using LabelEncoder

Scaled features using StandardScaler

Class Imbalance Handling

Applied SMOTE to balance target classes

Model Compilation

Optimizer: AdamW with learning_rate=0.0005

Loss: Binary Crossentropy

Metrics: Accuracy

Training

Epochs: 10(depends on dataset)

Batch Size: 32

Evaluation

Accuracy, Precision, Confusion Matrix

Plots for Accuracy and Loss

📊 Results Test Accuracy: ~

Precision: ~

Visualizations include:

Accuracy vs Epochs

Loss vs Epochs

Confusion Matrix Heatmap

📁 Dataset Format: .xlsx file

Source: Kaggle and custom logic (plant health classified based on researched feature dependency)

Target Label: Plant Health (Binary)

🛠️ Libraries Used pandas, numpy, matplotlib, seaborn

scikit-learn, imblearn

tensorflow, keras

📌 Key Takeaways Effective handling of imbalanced data using SMOTE

Regularization with L2 and Dropout to prevent overfitting

Use of BatchNormalization for faster convergence

High-quality visualizations for insights

