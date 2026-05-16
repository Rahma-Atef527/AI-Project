import pandas as pd #raham
import numpy as np#raham
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.preprocessing import  LabelEncoder,StandardScaler #raham
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

#======Load data====== Rahma
train_data = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

#======Remove spaces====== Rahma
train_data.columns = train_data.columns.str.strip()
test.columns = test.columns.str.strip()

#======Data cleaning====== Rahma
train_data.drop_duplicates(inplace=True)
train_data.reset_index(drop=True, inplace=True)

#======Handling null values======
col_question=['workclass','occupation','native-country']
for col in col_question:
    train_data[col]=train_data[col].str.strip().replace('?',np.nan)
    test[col]=test[col].str.strip().replace('?',np.nan)
    mode_value=train_data[col].mode()[0]
    train_data[col]=train_data[col].fillna(mode_value)
    test[col]=test[col].fillna(mode_value)

#======Encoding======
#Map Income and Sex to binary integers
train_data['Income'] = train_data['Income'].str.strip().map({'<=50K': 0, '>50K': 1})
train_data['sex'] = train_data['sex'].str.strip().map({'Male': 1, 'Female': 0})
test['sex'] = test['sex'].str.strip().map({'Male': 1, 'Female': 0})

#======Label Encoder======
#Convert categorical columns to numerical values
categorical_cols = ['workclass', 'occupation', 'native-country', 'race', 'marital-status', 'education', 'relationship']
for col in categorical_cols:
    train_data[col] = train_data[col].str.strip().replace('?', 'Unknown')
    test[col] = test[col].astype(str).str.strip().replace('?', 'Unknown')
    le = LabelEncoder()
    full_labels = pd.concat([train_data[col], test[col]], axis=0).astype(str)
    le.fit(full_labels)
    train_data[col] = le.transform(train_data[col])
    test[col] = le.transform(test[col])

#======VISUALIZATION======
def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
plot_correlation_heatmap(train_data)

def plot_distributions(df, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(18, 5))
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
plot_distributions(train_data, ['age', 'education-num', 'hours-per-week'])

def plot_categorical_vs_income(df, cat_col):
    plt.figure(figsize=(12, 6))
    order = df[cat_col].value_counts().index
    sns.countplot(data=df, y=cat_col, hue='Income', palette='viridis', order=order)
    plt.title(f'Income Distribution by {cat_col}')
    plt.xlabel('Count')
    plt.show()
plot_categorical_vs_income(train_data, 'occupation')
plot_categorical_vs_income(train_data, 'workclass')

plt.figure(figsize=(12, 10))
country_order = train_data['native-country'].value_counts().index

sns.countplot(
    data=train_data,
    y='native-country',
    hue='Income',
    palette='viridis',
    order=country_order
)
plt.title('Income Distribution by native-country')
plt.xlabel('Count')
plt.ylabel('Native Country')
plt.legend(title='Income', loc='lower right')
plt.tight_layout()
plt.show()

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()
plot_outliers_box(train_data, 'age')

#====== 6. Feature Selection ======
#Drop columns
columns_to_drop = ['education','fnlwgt']
train_data.drop(columns=columns_to_drop, inplace=True,errors='ignore')
test.drop(columns=columns_to_drop, inplace=True,errors='ignore')
print("Columns successfully dropped!")
print("Remaining columns:", train_data.columns.tolist())

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
plot_correlation_heatmap(train_data)

#======Outlier Handling======
normal_outlier_cols = ['age', 'hours-per-week']
for col in normal_outlier_cols:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_data[col] = np.clip(train_data[col], lower_bound, upper_bound)
    test[col] = np.clip(test[col], lower_bound, upper_bound)

capital_cols = ['capital-gain', 'capital-loss']
for col in capital_cols:
    upper_bound = train_data[col].quantile(0.99)
    train_data[col] = np.clip(train_data[col], 0, upper_bound)
    test[col] = np.clip(test[col], 0, upper_bound)
print("Outliers handled correctly without destroying capital columns!")

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()
plot_outliers_box(train_data, 'age')

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()
plot_outliers_box(train_data, 'hours-per-week')

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()
plot_outliers_box(train_data, 'capital-gain')


def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()
plot_outliers_box(train_data, 'capital-loss')



#======split======
y_train= train_data['Income']
X_train = train_data.drop('Income', axis=1)
y_test = test['Income'].str.strip().map({'<=50K.': 0, '>50K.': 1})
X_test = test.drop('Income', axis=1)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

#======scaling======
scaler = StandardScaler()
numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

#======HYPERPARAMETERS======
#Decision Tree
dt_params = {
    'max_depth': [5,10,15,None],
    'min_samples_split': [2,5,10]
}
dt_grid = RandomizedSearchCV(DecisionTreeClassifier(),dt_params,n_iter=5,cv=3)
dt_grid.fit(X_train, y_train)

#Random Forest
rf_params = {
    'n_estimators': [50,100,200],
    'max_depth': [5,10,20,None],
    'min_samples_split': [2,5,10]
}
rf_grid = RandomizedSearchCV(RandomForestClassifier(random_state=42),rf_params,n_iter=5,cv=3)
rf_grid.fit(X_train, y_train)

#XGBoost
xgb_params = {
    'n_estimators': [50,100,200],
    'max_depth': [3,6,10],
    'learning_rate': [0.01,0.1,0.2]
}
xgb_grid = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'),xgb_params,n_iter=10,cv=3)
xgb_grid.fit(X_train, y_train)

#======modeling======

def train_model(X_train, y_train, X_test, y_test):
    #Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
    dt_test_acc = accuracy_score(y_test, dt_preds)
    dt_f1 = f1_score(y_test, dt_preds)

    print("=== Decision Tree ===")
    print(f"Train Accuracy: {dt_train_acc * 100:.2f}%")
    print(f"Test Accuracy: {dt_test_acc * 100:.2f}%")
    print(f"F1-Score: {dt_f1 * 100:.2f}%\n")
    print("Classification Report for Decision Tree:")
    print(classification_report(y_test, dt_preds))
    print("-" * 30)

    #Logistic Regression
    lr = LogisticRegression(C=0.1, solver='liblinear')
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_train_acc = accuracy_score(y_train, lr.predict(X_train))
    lr_test_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)

    print("=== Logistic Regression ===")
    print(f"Train Accuracy: {lr_train_acc * 100:.2f}%")
    print(f"Test Accuracy: {lr_test_acc * 100:.2f}%")
    print(f"F1-Score: {lr_f1 * 100:.2f}%\n")
    print("Classification Report for Logistic Regression:")
    print(classification_report(y_test, lr_preds))
    print("-" * 30)

    #Random Forest
    rf = rf_grid.best_estimator_
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)

    print("=== Random Forest ===")
    print(f"Train Accuracy: {rf_train_acc * 100:.2f}%")
    print(f"Test Accuracy: {rf_test_acc * 100:.2f}%")
    print(f"F1-Score: {rf_f1 * 100:.2f}%\n")
    print("Classification Report for Random Forest:")
    print(classification_report(y_test, rf_preds))
    print("-" * 30)

    #XGBoost Classifier
    print("Training XGBoost... Please wait...")
    xgb_model = xgb_grid.best_estimator_
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
    xgb_test_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds)

    print("=== XGBoost ===")
    print(f"Train Accuracy: {xgb_train_acc * 100:.2f}%")
    print(f"Test Accuracy: {xgb_test_acc * 100:.2f}%")
    print(f"F1-Score: {xgb_f1 * 100:.2f}%\n")
    print("Classification Report for XGBoost:")
    print(classification_report(y_test, xgb_preds))
    print("-" * 30)

    #SVM
    print("Training SVM... Please wait...")
    svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_train_acc = accuracy_score(y_train, svm_model.predict(X_train))
    svm_test_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds)

    print("=== SVM ===")
    print(f"Train Accuracy: {svm_train_acc * 100:.2f}%")
    print(f"Test Accuracy: {svm_test_acc * 100:.2f}%")
    print(f"F1-Score: {svm_f1 * 100:.2f}%\n")
    print("Classification Report for SVM:")
    print(classification_report(y_test, svm_preds))
    print("-" * 30)

train_model(X_train, y_train, X_test, y_test)