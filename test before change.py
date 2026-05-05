import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelEncoder,StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
#load data

train_data = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
train_data.columns = train_data.columns.str.strip()
test.columns = test.columns.str.strip()

#data cleaning
train_data.drop_duplicates(inplace=True)
train_data.reset_index(drop=True, inplace=True)


#null values
col_question=['workclass','occupation','native-country']

for col in col_question:
    train_data[col]=train_data[col].str.strip().replace('?',np.nan)
    test[col]=test[col].str.strip().replace('?',np.nan)
    mode_value=train_data[col].mode()[0]
    train_data[col]=train_data[col].fillna(mode_value)
    test[col]=test[col].fillna(mode_value)

#encoding
train_data['Income'] = train_data['Income'].str.strip().map({'<=50K': 0, '>50K': 1})

train_data['sex'] = train_data['sex'].str.strip().map({'Male': 1, 'Female': 0})
test['sex'] = test['sex'].str.strip().map({'Male': 1, 'Female': 0})



# 1. قائمة الأعمدة التي سنحولها لأرقام
categorical_cols = ['workclass', 'occupation', 'native-country', 'race', 'marital-status', 'education', 'relationship']

# 2. تطبيق Label Encoder
le = LabelEncoder()

for col in categorical_cols:
    # تنظيف المسافات في التدريب والاختبار
    train_data[col] = train_data[col].str.strip().replace('?', 'Unknown')
    test[col] = test[col].astype(str).str.strip().replace('?', 'Unknown')

    # التدريب على train_data وتطبيقه على الاثنين
    le.fit(train_data[col])
    train_data[col] = le.transform(train_data[col])

    # تحويل بيانات الاختبار بناءً على ما تعلمه من التدريب
    # استخدام map للتعامل مع أي فئة جديدة قد تظهر في الاختبار
    test[col] = test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

def plot_correlation_heatmap(df):
    # نختار الأعمدة الرقمية فقط
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
    # ترتيب الفئات حسب الأكثر تكراراً
    order = df[cat_col].value_counts().index
    sns.countplot(data=df, y=cat_col, hue='Income', palette='viridis', order=order)
    plt.title(f'Income Distribution by {cat_col}')
    plt.xlabel('Count')
    plt.show()


plot_categorical_vs_income(train_data, 'occupation')
plot_categorical_vs_income(train_data, 'workclass')


# 3. إعداد الرسم البياني
plt.figure(figsize=(12, 10))

# تحديد ترتيب الدول بناءً على الأكثر تكراراً
country_order = train_data['native-country'].value_counts().index

# رسم العلاقة باستخدام Countplot
sns.countplot(
    data=train_data,
    y='native-country',
    hue='Income',
    palette='viridis',
    order=country_order
)

# إضافة التنسيقات النهائية
plt.title('Income Distribution by native-country')
plt.xlabel('Count')
plt.ylabel('Native Country')
plt.legend(title='Income', loc='lower right')

# تحسين المسافات لضمان عدم تداخل الأسماء
plt.tight_layout()

# عرض الرسمة
plt.show()

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()

plot_outliers_box(train_data, 'age')

# تحديد قائمة الأعمدة التي نريد حذفها
columns_to_drop = [
    'workclass',
    'education',
    'occupation',
    'race',
    'native-country',
    'fnlwgt'
]

# تنفيذ عملية الحذف (Drop) على بيانات التدريب والاختبار
train_data.drop(columns=columns_to_drop, inplace=True)
test.drop(columns=columns_to_drop, inplace=True)

# للتاكد من أن الأعمدة تم حذفها بنجاح
print("Columns successfully dropped!")
print("Remaining columns:", train_data.columns.tolist())

def plot_correlation_heatmap(df):
    # نختار الأعمدة الرقمية فقط
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

plot_correlation_heatmap(train_data)


# 1. الأعمدة العادية (نستخدم معاها طريقة الـ IQR بآمان)
normal_outlier_cols = ['age', 'hours-per-week']
for col in normal_outlier_cols:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_data[col] = np.clip(train_data[col], lower_bound, upper_bound)
    test[col] = np.clip(test[col], lower_bound, upper_bound)

# 2. أعمدة الأموال (نستخدم معاها 99th Percentile عشان نحافظ على الأرقام الحقيقية)
capital_cols = ['capital-gain', 'capital-loss']
for col in capital_cols:
    # نحسب الحد الأقصى بناءً على أعلى 1% من البيانات فقط
    upper_bound = train_data[col].quantile(0.99)
    # نقوم بالتقريب (أقل قيمة ستظل صفر، وأعلى قيمة ستكون الحد الأقصى)
    train_data[col] = np.clip(train_data[col], 0, upper_bound)
    test[col] = np.clip(test[col], 0, upper_bound)

print("Outliers handled correctly without destroying capital columns!")

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()

plot_outliers_box(train_data, 'age')

print("\nOutliers have been capped using Z-score method.")

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()

plot_outliers_box(train_data, 'hours-per-week')

print("\nOutliers have been capped using Z-score method.")

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()

plot_outliers_box(train_data, 'capital-gain')

print("\nOutliers have been capped using Z-score method.")

def plot_outliers_box(df, numeric_col):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Income', y=numeric_col, data=df, palette='Set2')
    plt.title(f'Outliers in {numeric_col} relative to Income')
    plt.show()

plot_outliers_box(train_data, 'capital-loss')

print("\nOutliers have been capped using Z-score method.")


#split
y_train= train_data['Income']
X_train = train_data.drop('Income', axis=1)

y_test = test['Income'].str.strip().map({'<=50K.': 0, '>50K.': 1})
X_test = test.drop('Income', axis=1)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# تأكدي من استخدام StandardScaler بدلاً من MinMaxScaler
scaler = StandardScaler()

numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

#======================modeling===================================#

def train_model(X_train, y_train, X_test, y_test):

    # 🔹 Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)

    dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
    dt_test_acc = accuracy_score(y_test, dt.predict(X_test))

    print("=== Decision Tree ===")
    print(f"Train Accuracy: {dt_train_acc*100:.2f}")
    print(f"Test Accuracy: {dt_test_acc*100:.2f}\n")

    print("\nClassification Report for Decision Tree:")
    print(classification_report(y_test, dt.predict(X_test)))

    # 🔹 Logistic Regression
    lr = LogisticRegression(C=0.1, solver='liblinear')
    lr.fit(X_train, y_train)

    lr_train_acc = accuracy_score(y_train, lr.predict(X_train))
    lr_test_acc = accuracy_score(y_test, lr.predict(X_test))

    print("=== Logistic Regression ===")
    print(f"Train Accuracy: {lr_train_acc*100:.2f}")
    print(f"Test Accuracy: {lr_test_acc*100:.2f}\n")

    print("\nClassification Report for Logistic Regression:")
    print(classification_report(y_test, lr.predict(X_test)))

    # 🔹 Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)

    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

    print("=== Random Forest ===")
    print(f"Train Accuracy: {rf_train_acc * 100:.2f}")
    print(f"Test Accuracy: {rf_test_acc * 100:.2f}\n")

    print("\nClassification Report for Random Forest:")
    print(classification_report(y_test, rf.predict(X_test)))
    # 🔹 XGBoost Classifier
    print("Training XGBoost... Please wait...")

    xgb_model = XGBClassifier(
        n_estimators=100,  # عدد الأشجار
        max_depth=6,  # عمق كل شجرة
        learning_rate=0.1,  # سرعة التعلم
        random_state=42,
        eval_metric='logloss'  # عشان يمنع أي رسائل تحذير (Warnings)
    )

    xgb_model.fit(X_train, y_train)

    xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
    xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))

    print("=== XGBoost ===")
    print(f"Train Accuracy: {xgb_train_acc * 100:.2f}")
    print(f"Test Accuracy: {xgb_test_acc * 100:.2f}")

    print("\nClassification Report for XGBoost:")
    print(classification_report(y_test, xgb_model.predict(X_test)))

    # 🔹 SVM
    print("Training SVM... Please wait...")

    # إضافة البارامتر هنا لموازنة الفئات تلقائياً
    svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)

    svm_model.fit(X_train, y_train)

    svm_train_acc = accuracy_score(y_train, svm_model.predict(X_train))
    svm_test_acc = accuracy_score(y_test, svm_model.predict(X_test))

    print("=== svm ===")
    print(f"Train Accuracy: {svm_train_acc * 100:.2f}")
    print(f"Test Accuracy: {svm_test_acc * 100:.2f}")

    print("\nClassification Report for SVM:")
    print(classification_report(y_test, svm_model.predict(X_test)))


train_model(X_train, y_train,X_test,y_test)