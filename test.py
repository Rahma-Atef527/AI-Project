import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# =========================
# Load Data
# =========================
train_data = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

train_data.columns = train_data.columns.str.strip()
test.columns = test.columns.str.strip()

# =========================
# Cleaning
# =========================
print(f"Number of duplicate rows in Train: {train_data.duplicated().sum()}\n")
train_data.drop_duplicates(inplace=True)
train_data.reset_index(drop=True, inplace=True)

# Missing values
col_question = ['workclass', 'occupation', 'native-country']

for col in col_question:
    train_data[col] = train_data[col].str.strip().replace('?', np.nan)
    test[col] = test[col].str.strip().replace('?', np.nan)

    mode_value = train_data[col].mode()[0]
    train_data[col] = train_data[col].fillna(mode_value)
    test[col] = test[col].fillna(mode_value)

# =========================
# Encoding
# =========================
train_data['Income'] = train_data['Income'].str.strip().map({'<=50K': 0, '>50K': 1})

train_data['sex'] = train_data['sex'].str.strip().map({'Male': 1, 'Female': 0})
test['sex'] = test['sex'].str.strip().map({'Male': 1, 'Female': 0})

categorical_cols = ['workclass', 'occupation', 'native-country', 'race', 'marital-status', 'education', 'relationship']

train_data = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# =========================
# Outliers
# =========================
normal_outlier_cols = ['age', 'hours-per-week']

for col in normal_outlier_cols:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    train_data[col] = np.clip(train_data[col], lower, upper)
    test[col] = np.clip(test[col], lower, upper)

capital_cols = ['capital-gain', 'capital-loss']

for col in capital_cols:
    upper = train_data[col].quantile(0.99)

    train_data[col] = np.clip(train_data[col], 0, upper)
    test[col] = np.clip(test[col], 0, upper)

# =========================
# Split
# =========================
y_train = train_data['Income']
X_train = train_data.drop('Income', axis=1)

y_test = test['Income'].str.strip().map({'<=50K.': 0, '>50K.': 1})
X_test = test.drop('Income', axis=1)

# align columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# =========================
# Scaling
# =========================
scaler = MinMaxScaler()
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# =========================
# Training Function
# =========================
def train_model(X_train, y_train, X_test, y_test):

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)

    print("\n=== Decision Tree ===")
    print("Test Accuracy:", accuracy_score(y_test, dt.predict(X_test)))
    print(classification_report(y_test, dt.predict(X_test)))

    # Logistic Regression
    lr = LogisticRegression(C=0.1, solver='liblinear', class_weight='balanced')
    lr.fit(X_train, y_train)

    print("\n=== Logistic Regression ===")
    print("Test Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
    print(classification_report(y_test, lr.predict(X_test)))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    print("\n=== Random Forest ===")
    print("Test Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
    print(classification_report(y_test, rf.predict(X_test)))

    # =========================
    # XGBoost (Improved)
    # =========================
    print("\nTraining XGBoost...")

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)

    # Cross Validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1')
    print("\nCross Validation F1:", cv_scores)
    print("Mean F1:", cv_scores.mean())

    # Threshold Tuning
    y_probs = xgb_model.predict_proba(X_test)[:, 1]

    threshold = 0.6
    y_pred = (y_probs > threshold).astype(int)

    print("\n=== XGBoost ===")
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # SVM
    print("\nTraining SVM...")

    svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    svm_model.fit(X_train, y_train)

    print("\n=== SVM ===")
    print("Test Accuracy:", accuracy_score(y_test, svm_model.predict(X_test)))
    print(classification_report(y_test, svm_model.predict(X_test)))


# =========================
# Run
# =========================
train_model(X_train, y_train, X_test, y_test)