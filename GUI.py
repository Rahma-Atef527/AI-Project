import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
import threading

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score


# THEME
# ─────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#0F172A"
CARD = "#1E293B"
PURPLE = "#6C63FF"
GREEN = "#10B981"
RED = "#EF4444"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"


# DATA URLs
# ─────────────────────────────────────────────────────────
TRAIN_URL = "https://raw.githubusercontent.com/Rahma-Atef527/AI-Project/main/train_data.csv"
TEST_URL  = "https://raw.githubusercontent.com/Rahma-Atef527/AI-Project/main/test_data.csv"


# OPTIONS
# ─────────────────────────────────────────────────────────
WORKCLASSES = [
    "Private","Self-emp-not-inc","Self-emp-inc",
    "Federal-gov","Local-gov","State-gov"
]

EDUCATIONS = [
    "Bachelors","Some-college","HS-grad",
    "Masters","Doctorate","Assoc-voc"
]

MARITAL_STATUS = [
    "Never-married","Married-civ-spouse",
    "Divorced","Separated"
]

OCCUPATIONS = [
    "Tech-support","Sales","Exec-managerial",
    "Prof-specialty","Adm-clerical"
]

RELATIONSHIPS = [
    "Not-in-family","Husband","Wife",
    "Own-child","Unmarried"
]

RACES = [
    "White","Black","Asian-Pac-Islander","Other"
]

COUNTRIES = [
    "United-States","India","Canada",
    "Germany","England","Other"
]


# MODEL
# ─────────────────────────────────────────────────────────
class IncomeModel:

    FEATURE_COLS = [
        "age","workclass","education","education-num",
        "marital-status","occupation","relationship",
        "race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country"
    ]

    CAT_COLS = [
        "workclass","education","marital-status",
        "occupation","relationship","race","native-country"
    ]

    NUM_COLS = [
        "age","education-num","capital-gain",
        "capital-loss","hours-per-week"
    ]

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.trained = False
        self.accuracy = 0
        self.f1 = 0

    def _encode(self, df, fit=False):

        df = df.copy()

        for col in self.CAT_COLS:

            df[col] = df[col].astype(str).str.strip()

            if fit:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col])

            known = set(self.encoders[col].classes_)

            df[col] = df[col].apply(
                lambda x: x if x in known
                else self.encoders[col].classes_[0]
            )

            df[col] = self.encoders[col].transform(df[col])

        return df

    def train(self, train_df, test_df):

        train_df = self._encode(train_df, fit=True)
        test_df  = self._encode(test_df)

        X_train = train_df[self.FEATURE_COLS]
        y_train = train_df["Income"]

        X_test = test_df[self.FEATURE_COLS]
        y_test = test_df["Income"]

        X_train[self.NUM_COLS] = self.scaler.fit_transform(
            X_train[self.NUM_COLS]
        )

        X_test[self.NUM_COLS] = self.scaler.transform(
            X_test[self.NUM_COLS]
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, preds) * 100
        self.f1 = f1_score(y_test, preds) * 100

        self.trained = True

    def predict(self, row):

        df = pd.DataFrame([row])

        df = self._encode(df)

        X = df[self.FEATURE_COLS]

        X[self.NUM_COLS] = self.scaler.transform(
            X[self.NUM_COLS]
        )

        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0][1]

        return pred, proba



# LOAD DATA
# ─────────────────────────────────────────────────────────
def load_and_clean():

    train = pd.read_csv(TRAIN_URL)
    test  = pd.read_csv(TEST_URL)

    for df in [train, test]:
        df.columns = df.columns.str.strip()

    for col in ["workclass", "occupation", "native-country"]:

        for df in [train, test]:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("?", np.nan)
            )

        mode_val = train[col].mode()[0]

        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)

    train["Income"] = train["Income"].str.strip().map({
        "<=50K":0,
        ">50K":1
    })

    test["Income"] = test["Income"].str.strip().map({
        "<=50K.":0,
        ">50K.":1,
        "<=50K":0,
        ">50K":1
    })

    train["sex"] = train["sex"].map({
        "Male":1,
        "Female":0
    })

    test["sex"] = test["sex"].map({
        "Male":1,
        "Female":0
    })

    train.drop(columns=["fnlwgt"], inplace=True)
    test.drop(columns=["fnlwgt"], inplace=True)

    return train, test



# APP
# ─────────────────────────────────────────────────────────
class App(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.main = None
        self.geometry("1200x700")
        self.title("Income Classifier")
        self.configure(fg_color=BG)

        self.model_obj = IncomeModel()

        self.build_ui()

        threading.Thread(
            target=self.train_model,
            daemon=True
        ).start()


    # TRAIN
    # ─────────────────────────────────────────
    def train_model(self):

        try:

            train_df, test_df = load_and_clean()

            self.model_obj.train(train_df, test_df)

            self.after(
                0,
                lambda:
                self.status.configure(
                    text="✅ Model Ready"
                )
            )

        except Exception as e:

            self.after(
                0,
                lambda:
                self.status.configure(
                    text=f"❌ {e}"
                )
            )


    # UI
    # ─────────────────────────────────────────
    def build_ui(self):

        # MAIN
        self.main = ctk.CTkFrame(
            self,
            fg_color=BG
        )

        self.main.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        # LEFT
        self.left = ctk.CTkFrame(
            self.main,
            fg_color=BG
        )

        self.left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0,10)
        )

        # RIGHT
        self.right = ctk.CTkFrame(
            self.main,
            fg_color=CARD,
            width=320,
            corner_radius=20
        )

        self.right.pack(
            side="right",
            fill="y"
        )

        # TITLE
        ctk.CTkLabel(
            self.left,
            text="💰 Income Classifier",
            font=("Segoe UI", 28, "bold"),
            text_color=TEXT
        ).pack(anchor="w", pady=(10,5))

        ctk.CTkLabel(
            self.left,
            text="Predict income using Machine Learning",
            font=("Segoe UI", 14),
            text_color=MUTED
        ).pack(anchor="w")

        # STATUS
        self.status = ctk.CTkLabel(
            self.left,
            text="⏳ Training model...",
            font=("Segoe UI", 14),
            text_color="#FBBF24"
        )

        self.status.pack(anchor="w", pady=(10,20))

        # FORM
        form = ctk.CTkFrame(
            self.left,
            fg_color=CARD,
            corner_radius=20
        )

        form.pack(fill="both", expand=True)

        # VARIABLES
        self.age_var = ctk.StringVar()
        self.hrs_var = ctk.StringVar()
        self.edu_var = ctk.StringVar()
        self.cap_var = ctk.StringVar()

        self.wc_var = ctk.StringVar(value=WORKCLASSES[0])
        self.ed_var = ctk.StringVar(value=EDUCATIONS[0])
        self.ms_var = ctk.StringVar(value=MARITAL_STATUS[0])
        self.oc_var = ctk.StringVar(value=OCCUPATIONS[0])
        self.re_var = ctk.StringVar(value=RELATIONSHIPS[0])
        self.ra_var = ctk.StringVar(value=RACES[0])
        self.co_var = ctk.StringVar(value=COUNTRIES[0])
        self.sx_var = ctk.StringVar(value="Male")

        # GRID
        fields = [

            ("👤 Age", self.age_var),
            ("⏰ Hours/week", self.hrs_var),

            ("🎓 Education Years", self.edu_var),
            ("💵 Capital Gain", self.cap_var)
        ]

        row = 0
        col = 0

        for label, var in fields:

            box = ctk.CTkFrame(
                form,
                fg_color="transparent"
            )

            box.grid(
                row=row,
                column=col,
                padx=10,
                pady=10,
                sticky="ew"
            )

            ctk.CTkLabel(
                box,
                text=label,
                font=("Segoe UI", 13)
            ).pack(anchor="w")

            ctk.CTkEntry(
                box,
                textvariable=var,
                height=40,
                corner_radius=12
            ).pack(fill="x", pady=5)

            col += 1

            if col > 1:
                col = 0
                row += 1

        # DROPDOWNS
        dropdowns = [

            ("💼 Workclass", self.wc_var, WORKCLASSES),
            ("📘 Education", self.ed_var, EDUCATIONS),

            ("💍 Marital", self.ms_var, MARITAL_STATUS),
            ("🛠 Occupation", self.oc_var, OCCUPATIONS),

            ("👨 Relationship", self.re_var, RELATIONSHIPS),
            ("🌍 Country", self.co_var, COUNTRIES),

            ("🧬 Race", self.ra_var, RACES),
            ("⚧ Sex", self.sx_var, ["Male","Female"])
        ]

        for text, var, values in dropdowns:

            box = ctk.CTkFrame(
                form,
                fg_color="transparent"
            )

            box.grid(
                row=row,
                column=col,
                padx=10,
                pady=10,
                sticky="ew"
            )

            ctk.CTkLabel(
                box,
                text=text,
                font=("Segoe UI", 13)
            ).pack(anchor="w")

            ctk.CTkComboBox(
                box,
                values=values,
                variable=var,
                height=40,
                corner_radius=12
            ).pack(fill="x", pady=5)

            col += 1

            if col > 1:
                col = 0
                row += 1

        # BUTTONS
        btns = ctk.CTkFrame(
            self.left,
            fg_color="transparent"
        )

        btns.pack(fill="x", pady=15)

        self.predict_btn = ctk.CTkButton(
            btns,
            text="Predict Income",
            height=50,
            corner_radius=15,
            fg_color=PURPLE,
            hover_color="#5B54D6",
            font=("Segoe UI", 15, "bold"),
            command=self.predict
        )

        self.predict_btn.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0,10)
        )

        ctk.CTkButton(
            btns,
            text="Reset",
            height=50,
            corner_radius=15,
            fg_color="#334155",
            hover_color="#475569",
            command=self.reset
        ).pack(side="left")

        # RIGHT PANEL
        ctk.CTkLabel(
            self.right,
            text="Prediction Result",
            font=("Segoe UI", 24, "bold")
        ).pack(pady=20)

        self.result_label = ctk.CTkLabel(
            self.right,
            text="Waiting...",
            font=("Segoe UI", 40, "bold"),
            text_color=MUTED
        )

        self.result_label.pack(pady=25)

        self.prob_label = ctk.CTkLabel(
            self.right,
            text="0%",
            font=("Segoe UI", 30)
        )

        self.prob_label.pack()

        self.progress = ctk.CTkProgressBar(
            self.right,
            width=250,
            progress_color=GREEN
        )

        self.progress.pack(pady=20)

        self.progress.set(0)

        self.acc_label = ctk.CTkLabel(
            self.right,
            text="Accuracy: --"
        )

        self.acc_label.pack(pady=5)

        self.f1_label = ctk.CTkLabel(
            self.right,
            text="F1 Score: --"
        )

        self.f1_label.pack()


    # PREDICT
    # ─────────────────────────────────────────
    def predict(self):

        if not self.model_obj.trained:

            messagebox.showinfo(
                "Wait",
                "Model still training"
            )

            return

        try:

            row = {

                "age": int(self.age_var.get()),
                "workclass": self.wc_var.get(),
                "education": self.ed_var.get(),
                "education-num": int(self.edu_var.get()),
                "marital-status": self.ms_var.get(),
                "occupation": self.oc_var.get(),
                "relationship": self.re_var.get(),
                "race": self.ra_var.get(),
                "sex": 1 if self.sx_var.get()=="Male" else 0,
                "capital-gain": int(self.cap_var.get()),
                "capital-loss": 0,
                "hours-per-week": int(self.hrs_var.get()),
                "native-country": self.co_var.get()
            }

            pred, proba = self.model_obj.predict(row)

            label = ">50K" if pred == 1 else "≤50K"

            color = GREEN if pred == 1 else RED

            self.result_label.configure(
                text=label,
                text_color=color
            )

            self.prob_label.configure(
                text=f"{round(proba*100)}%"
            )

            self.progress.set(proba)

            self.acc_label.configure(
                text=f"Accuracy: {self.model_obj.accuracy:.2f}%"
            )

            self.f1_label.configure(
                text=f"F1 Score: {self.model_obj.f1:.2f}%"
            )

        except:

            messagebox.showerror(
                "Error",
                "Please enter valid numbers"
            )


    # RESET
    # ─────────────────────────────────────────
    def reset(self):

        self.age_var.set("")
        self.hrs_var.set("")
        self.edu_var.set("")
        self.cap_var.set("")

        self.result_label.configure(
            text="Waiting...",
            text_color=MUTED
        )

        self.prob_label.configure(
            text="0%"
        )

        self.progress.set(0)



# RUN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    app = App()

    app.mainloop()