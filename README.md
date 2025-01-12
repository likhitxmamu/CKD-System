# Chronic Kidney Disease (CKD) Detection System

This project is a machine learning-based system designed to detect Chronic Kidney Disease (CKD) using patient data.

## Introduction

Chronic Kidney Disease is a condition characterized by a gradual loss of kidney function over time. Early detection is crucial for effective management and treatment. This project utilizes machine learning algorithms to analyze patient data and predict the presence of CKD.

## Features

- Data preprocessing and cleaning
- Exploratory data analysis
- Implementation of various machine learning models
- Model evaluation and selection
- User-friendly interface for predictions

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/likhitxmamu/CKD-System.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd CKD-System
   ```

3. **Ensure Python version is 3.11.5:**

   Make sure you have Python 3.11.5 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

4. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  
   # On Windows: venv\Scripts\activate
   ```

5. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**

   Ensure that the dataset is placed in the `data` directory.

2. **Train and preprocess model:**

   ```bash
   cd notebooks
   python inference.py
   ```

3. **Run the app:**

   ```bash
   cd Frontend
   python app.py
   ```

## Dataset

The dataset used for this project contains various medical attributes of patients, which are used as features for predicting CKD. Ensure that the dataset is properly formatted and placed in the `data` directory before running the preprocessing script.

## Model Training

The project explores multiple machine learning algorithms, including logistic regression, decision trees, and support vector machines. The training scripts are located in the `src` directory. After preprocessing the data, you can train the models using the `inference.py` script.

## Results

Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed results and model evaluations are documented in the `notebooks` directory, where Jupyter notebooks provide insights into the model selection process.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more details.

## Author

Likhit Kumar

---

*Note: This project is for educational purposes only and should not be used for medical diagnostics without proper validation and approval.*