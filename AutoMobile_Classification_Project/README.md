# 🚗 Automobile Origin Classification

A machine learning project that classifies the country of origin (USA, Europe, or Japan) of automobiles based on their technical specifications, using the classic Auto MPG dataset.

## 🎯 Objective

To build a classification model that predicts a car's origin (USA, Europe, or Japan) using features such as mpg, cylinders, displacement, horsepower, weight, acceleration, and model year.

## 📊 Dataset

- **File:** `automobile.csv`
- **Records:** 398 cars
- **Features:** `name`, `mpg`, `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin`
- **Target Variable:** `origin` (USA / Europe / Japan)

## 🛠️ Project Workflow

1. **Data Exploration** – Inspected dataset shape, structure, and summary statistics using `.info()` and `.describe()`.
2. **Missing Value Handling** – Filled missing `horsepower` values (6 missing entries) using the column mean.
3. **Duplicate Check** – Verified there were no duplicate records in the dataset.
4. **Feature Cleaning** – Dropped the non-predictive `name` column.
5. **Exploratory Data Analysis (EDA)** –
   - Visualized the distribution of cars by origin using a count plot.
   - Plotted a correlation heatmap to examine relationships between numeric features.
6. **Train-Test Split** – Split the data into 80% training and 20% testing sets.
7. **Feature Scaling** – Standardized features using `StandardScaler`.
8. **Model Training** – Trained a Logistic Regression classifier to predict car origin.
9. **Evaluation** – Assessed performance using accuracy score and a full classification report (precision, recall, F1-score).

## 📈 Results

- **Accuracy:** 82.5%
- The model performed strongly on **USA** (F1-score: 0.92), with comparatively lower performance on **Europe** and **Japan**, likely due to class imbalance (53 USA vs. 14 Europe vs. 13 Japan test samples).

## 🧠 Skills Demonstrated

- Multiclass classification
- Data cleaning and missing value imputation
- Exploratory Data Analysis (EDA) and correlation analysis
- Feature scaling
- Model evaluation using accuracy and classification reports

## 📦 Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python | Core programming language |
| `pandas`, `numpy` | Data manipulation and cleaning |
| `scikit-learn` | Feature scaling, model training, and evaluation |
| `matplotlib`, `seaborn` | EDA and visualization |

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project

1. Clone the repository:
```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
```
2. Run the notebook:
```bash
   jupyter notebook automobile_classification.ipynb
```

## 📁 Project Structure

├── automobile.csv                    # Dataset

├── automobile_classification.ipynb   # Main notebook with EDA, training, and evaluation

├── README.md                         # Project documentation

└── requirements.txt                   # Python dependencies

## 📌 Notes

- Class imbalance in the dataset (majority of cars are USA-made) affects recall for the Europe and Japan classes; techniques like class weighting or resampling could improve performance further.
- This project is intended for educational purposes to demonstrate multiclass classification workflows.
