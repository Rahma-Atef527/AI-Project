# نعمل رن للجزء ده الاول علشان نتاكد ان المكتبات شغالة و بعدين نعمله كومنت و نشغل اللى تحت

import pandas as pd
import seaborn as sns
import sklearn

print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("\n✅ All libraries are working perfectly!")

#نفك الكومنت من عليه و نعمله رن بعد ما نتاكد ان اللى فوق شغال و نعمل للى فوق كومنت
"""
import pandas as pd

# قراءة الداتا
train_df = pd.read_csv('train_data.csv')

# معرفة هل فيه قيم ناقصة؟
print("--- Missing Values in each column ---")
print(train_df.isnull().sum())
"""