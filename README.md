# SymptomSphere: AI-Powered Medical Training DSS

## Overview

SymptomSphere is an intelligent Decision Support System (DSS) designed to enhance medical training by improving diagnostic accuracy for medical students and professionals. Built for CSCN8030 – Artificial Intelligence for Business Decisions and Transformation (Team 4), it leverages a Random Forest Classifier to predict medical conditions (Diabetes, Pneumonia, Cancer) using the [Medical Condition Prediction Dataset](https://www.kaggle.com/datasets/marius2303/medical-condition-prediction-dataset/data). The system provides actionable feedback, addresses dataset imbalances, and supports scalable, realistic training scenarios.

**Key Features**:
- **Data Analysis**: Visualizes condition distributions, age trends, and clinical feature patterns.
- **Predictive Model**: Robust classification with feature importance and probability outputs.
- **Educational Feedback**: Simulates diagnostics with confidence scores for training.
- **Stakeholder Benefits**: Enhances student confidence, reduces instructor workload, and supports institutional scalability.

## Dataset

- **Source**: Synthetic patient records from Kaggle (10,000 entries).
- **Features**:
  - `id`, `full_name`: Identifiers.
  - `age`, `bmi`, `blood_pressure`, `glucose_levels`: Numeric clinical features.
  - `gender` (Male, Female, Non-Binary), `smoking_status` (Smoker, Non-Smoker, Former-Smoker): Categorical features.
  - `condition`: Target (Diabetes: 60.13%, Pneumonia: 25.27%, Cancer: 14.60%).
- **Preprocessing**: Median imputation for missing values, case-normalized label encoding for categoricals.

Download the dataset and save it as `medical_conditions_dataset.csv` in the project root.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/symptomsphere-dss.git
   cd symptomsphere-dss
   ```

2. **Set Up Virtual Environment** (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` contents:
   ```
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook "Project’s data analysis.ipynb"
   ```

## Usage

1. **Run the Notebook**:
   - Execute cells sequentially to:
     - Load and preprocess the dataset.
     - Train a Random Forest Classifier (300 estimators, balanced weights).
     - Visualize condition distributions, confusion matrices, and feature importance.
   - Outputs include classification reports and plots for stakeholder insights.

2. **Make Predictions**:
   - Use the `predict_patient` function for new patient diagnostics:
     ```python
     _ = predict_patient(
         age=50.0,
         gender='male',
         smoking_status='non-smoker',
         bmi=25.0,
         blood_pressure=120.0,
         glucose=150.0
     )
     ```
   - Example output:
     ```
     Predicted condition for new patient: Diabetic
     Class probabilities (descending):
       Diabetic: 0.653
       Pneumonia: 0.290
       Cancer: 0.057
     ```

3. **Customize**:
   - Adjust hyperparameters (e.g., `n_estimators`) for performance.
   - Integrate with conversational AI or add SMOTE for handling rare classes like Cancer.

## Project Structure

- `Project’s data analysis.ipynb`: Core notebook with analysis, modeling, and predictions.
- `medical_conditions_dataset.csv`: Dataset (download from Kaggle).
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Results and Insights

- **Model Performance**: High recall for Diabetes due to its prevalence; Cancer prediction needs improvement via oversampling.
- **Key Findings**:
  - Glucose levels are critical for Diabetes, age for Cancer, and blood pressure for differentiation.
  - Dataset imbalance (60.13% Diabetes) is mitigated with balanced weights.
- **DSS Impact**:
  - **Students**: Gain realistic diagnostic practice with instant feedback.
  - **Instructors**: Track progress and reduce grading effort.
  - **Institutions**: Achieve cost-effective, scalable training aligned with accreditation.

## Future Enhancements

- Apply SMOTE for balancing rare classes (Cancer, Pneumonia).
- Integrate generative AI for synthetic case generation.
- Develop a web interface (e.g., Streamlit) for interactive diagnostics.

## Contributors

- Team 4 (CSCN8030)


## Acknowledgments

- **Dataset**: Marius2303 on Kaggle.
- **Libraries**: pandas, scikit-learn, seaborn, matplotlib.
- **Course**: CSCN8030, Artificial Intelligence for Business Decisions and Transformation.

For issues or contributions, please open an issue or pull request on GitHub.