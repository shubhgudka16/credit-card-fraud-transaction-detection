# 💳 Credit Card Fraud Detection using Logistic Regression

A Machine Learning project that detects fraudulent credit card transactions using **Logistic Regression**. This project demonstrates the complete machine learning workflow, from data preprocessing and handling imbalanced data to model training, evaluation, and prediction.

---

## 📖 Overview

Credit card fraud is a major concern in the banking and financial sector. The goal of this project is to build a binary classification model that can identify whether a credit card transaction is **legitimate** or **fraudulent**.

Since fraudulent transactions represent only a small fraction of all transactions, the dataset is highly imbalanced. To improve model performance, **Random Undersampling** is used to create a balanced dataset before training the model.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions.
* Handle class imbalance using Random Undersampling.
* Train a Logistic Regression model.
* Evaluate the model using multiple performance metrics.
* Predict whether a transaction is legitimate or fraudulent.

---

## 📂 Dataset

This project uses the **Credit Card Fraud Detection** dataset.

Due to GitHub's file size limitations, the dataset is **not included** in this repository.

### Download the Dataset

Download the dataset from Kaggle:

**https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud**

After downloading:

1. Extract the ZIP file.
2. Copy `creditcard.csv` into the project folder.

Your project structure should look like this:

```text
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── model.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn

---

## 📚 Machine Learning Concepts

* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Random Undersampling
* Train-Test Split
* Logistic Regression
* Binary Classification
* Model Evaluation

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

### 2. Navigate to the Project Folder

```bash
cd Credit-Card-Fraud-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the dataset from Kaggle and place `creditcard.csv` in the project folder.

### 5. Run the Project

```bash
python model.py
```

---

## 🚀 Project Workflow

* Load the dataset
* Display dataset information
* Check for missing values
* Analyze class distribution
* Separate legitimate and fraudulent transactions
* Create a balanced dataset using Random Undersampling
* Split features and target variable
* Perform Train-Test Split
* Train the Logistic Regression model
* Evaluate the model
* Predict new transactions

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

  * Precision
  * Recall
  * F1-Score

---

## 🧪 Testing the Model

Instead of manually entering all 30 feature values, simply enter the row number of any transaction from the dataset.

Example:

```text
Enter row number from the dataset: 100
```

Output:

```text
Actual Class: 0
Predicted: Legitimate Transaction
```

or

```text
Actual Class: 1
Predicted: Fraudulent Transaction
```

This method uses real transaction data from the dataset and makes testing easier.

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📦 Requirements

Create a `requirements.txt` file with the following libraries:

```text
numpy
pandas
scikit-learn
```

Or install them manually:

```bash
pip install numpy pandas scikit-learn
```

---

## 📈 Future Improvements

* Apply feature scaling for better performance.
* Compare Logistic Regression with other machine learning algorithms.
* Use SMOTE for handling imbalanced data.
* Perform hyperparameter tuning.
* Build a web application using Streamlit or Flask.
* Deploy the trained model to the cloud.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork this repository.
2. Create a new branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 👨‍💻 Author

**Shubh Gudka**

Diploma in Computer Engineering

