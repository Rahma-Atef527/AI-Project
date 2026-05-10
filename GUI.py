import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import threading

# ── Auto-install missing packages ─────────────────────────────────────────────
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import pandas as pd
except ImportError:
    install("pandas"); import pandas as pd

try:
    import numpy as np
except ImportError:
    install("numpy"); import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    install("scikit-learn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

# ── Data URLs ──────────────────────────────────────────────────────────────────
TRAIN_URL = "https://raw.githubusercontent.com/Rahma-Atef527/AI-Project/main/train_data.csv"
TEST_URL  = "https://raw.githubusercontent.com/Rahma-Atef527/AI-Project/main/test_data.csv"

# ── Dropdown options ───────────────────────────────────────────────────────────
WORKCLASSES    = ["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov","State-gov","Without-pay","Never-worked"]
EDUCATIONS     = ["Bachelors","Some-college","11th","HS-grad","Prof-school","Assoc-acdm","Assoc-voc","9th","7th-8th","12th","Masters","1st-4th","10th","Doctorate","5th-6th","Preschool"]
MARITAL_STATUS = ["Married-civ-spouse","Divorced","Never-married","Separated","Widowed","Married-spouse-absent","Married-AF-spouse"]
OCCUPATIONS    = ["Tech-support","Craft-repair","Other-service","Sales","Exec-managerial","Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical","Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"]
RELATIONSHIPS  = ["Wife","Own-child","Husband","Not-in-family","Other-relative","Unmarried"]
RACES          = ["White","Asian-Pac-Islander","Amer-Indian-Eskimo","Other","Black"]
COUNTRIES      = ["United-States","Cuba","Jamaica","India","Mexico","Japan","Greece","Italy","Canada","China","Iran","England","Germany","Vietnam","Philippines","Poland","Other"]

# ── Colors ─────────────────────────────────────────────────────────────────────
BG         = "#F8F8F8"
CARD_BG    = "#FFFFFF"
BORDER     = "#E0E0E0"
PURPLE     = "#7F77DD"
GREEN      = "#1D9E75"
RED        = "#E24B4A"
AMBER      = "#BA7517"
TEXT_MAIN  = "#1A1A1A"
TEXT_MUTED = "#6B6B6B"
TEXT_LABEL = "#444444"


# ── Model ──────────────────────────────────────────────────────────────────────
class IncomeModel:
    def __init__(self):
        self.model    = None
        self.scaler   = StandardScaler()
        self.encoders = {}
        self.trained  = False
        self.accuracy = 0.0
        self.f1       = 0.0

    FEATURE_COLS = ["age","workclass","education","education-num","marital-status",
                    "occupation","relationship","race","sex","capital-gain",
                    "capital-loss","hours-per-week","native-country"]
    CAT_COLS     = ["workclass","education","marital-status","occupation",
                    "relationship","race","native-country"]
    NUM_COLS     = ["age","education-num","capital-gain","capital-loss","hours-per-week"]

    def _encode(self, df, fit=False):
        df = df.copy()
        for col in self.CAT_COLS:
            df[col] = df[col].astype(str).str.strip()
            if fit:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col])
            known = set(self.encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known else self.encoders[col].classes_[0])
            df[col] = self.encoders[col].transform(df[col])
        return df

    def train(self, train_df, test_df):
        train_df = self._encode(train_df, fit=True)
        test_df  = self._encode(test_df)

        X_tr = train_df[self.FEATURE_COLS].copy()
        y_tr = train_df["Income"]
        X_te = test_df[self.FEATURE_COLS].copy()
        y_te = test_df["Income"]

        X_tr[self.NUM_COLS] = self.scaler.fit_transform(X_tr[self.NUM_COLS])
        X_te[self.NUM_COLS] = self.scaler.transform(X_te[self.NUM_COLS])

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_tr, y_tr)

        preds = self.model.predict(X_te)
        self.accuracy = accuracy_score(y_te, preds) * 100
        self.f1       = f1_score(y_te, preds) * 100
        self.trained  = True

    def predict(self, row: dict):
        df = pd.DataFrame([row])
        df = self._encode(df)
        X  = df[self.FEATURE_COLS].copy()
        X[self.NUM_COLS] = self.scaler.transform(X[self.NUM_COLS])
        pred  = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0][1]
        return int(pred), float(proba)


def load_and_clean(url_train, url_test):
    train = pd.read_csv(url_train)
    test  = pd.read_csv(url_test)
    for df in [train, test]:
        df.columns = df.columns.str.strip()
    train.drop_duplicates(inplace=True)

    for col in ["workclass", "occupation", "native-country"]:
        for df in [train, test]:
            df[col] = df[col].astype(str).str.strip().replace("?", np.nan)
        mode_val = train[col].mode()[0]
        train[col] = train[col].fillna(mode_val)
        test[col]  = test[col].fillna(mode_val)

    train["Income"] = train["Income"].str.strip().map({"<=50K": 0, ">50K": 1})
    train["sex"]    = train["sex"].str.strip().map({"Male": 1, "Female": 0})
    test["sex"]     = test["sex"].str.strip().map({"Male": 1, "Female": 0})
    test["Income"]  = test["Income"].str.strip().map({"<=50K.": 0, ">50K.": 1,
                                                       "<=50K":  0, ">50K":  1})
    for df in [train, test]:
        df.drop(columns=["fnlwgt"], errors="ignore", inplace=True)

    return train, test


# ── GUI ────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Income Classifier")
        self.geometry("700x840")
        self.resizable(True, True)
        self.configure(bg=BG)

        self.model_obj = IncomeModel()
        self._build_ui()

        # Train in background so window appears immediately
        t = threading.Thread(target=self._train_thread, daemon=True)
        t.start()

    # ── Background training ──────────────────────────────────────────────────
    def _train_thread(self):
        try:
            train_df, test_df = load_and_clean(TRAIN_URL, TEST_URL)
            self.model_obj.train(train_df, test_df)
            msg = "✅  Model ready and loaded successfully"  # <--- Simplified message
            self.after(0, lambda: self._set_status(msg, "#E8F5E9", GREEN, enable=True))
        except Exception as ex:
            msg = f"❌  Error: {ex}"
            self.after(0, lambda: self._set_status(msg, "#FFEBEE", RED, enable=False))

    def _set_status(self, msg, bg, fg, enable):
        self.status_var.set(msg)
        self.status_lbl.configure(bg=bg, fg=fg)
        if enable:
            self.predict_btn.configure(state="normal")

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.main = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=self.main, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        self.main.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Title
        tk.Label(self.main, text="Income Classifier", font=("Helvetica", 18, "bold"),
                 bg=BG, fg=TEXT_MAIN).pack(anchor="w", padx=20, pady=(18, 2))
        tk.Label(self.main, text="Predict whether income is  >50K  or  ≤50K",
                 font=("Helvetica", 11), bg=BG, fg=TEXT_MUTED).pack(anchor="w", padx=20, pady=(0, 10))

        # Status bar
        self.status_var = tk.StringVar(value="⏳  Loading data from GitHub and training model…  please wait")
        self.status_lbl = tk.Label(self.main, textvariable=self.status_var,
                                   font=("Helvetica", 10), bg="#FFF8E1", fg=AMBER,
                                   relief="flat", anchor="w", padx=12, pady=7)
        self.status_lbl.pack(fill="x", padx=20, pady=(0, 12))

        # ── Numeric card ──
        self._section("NUMERIC FEATURES")
        num_card = self._card()
        g = tk.Frame(num_card, bg=CARD_BG)
        g.pack(fill="x", padx=14, pady=10)
        g.columnconfigure(0, weight=1, uniform="c")
        g.columnconfigure(1, weight=1, uniform="c")

        self.age_var = tk.StringVar()
        self.hrs_var = tk.StringVar()
        self.edu_var = tk.StringVar()
        self.cap_var = tk.StringVar()

        self._textbox(g, "Age",              self.age_var, 0, 0, "17 – 90")
        self._textbox(g, "Hours per week",   self.hrs_var, 0, 1, "1 – 99")
        self._textbox(g, "Education years",  self.edu_var, 1, 0, "1 – 16")
        self._textbox(g, "Capital gain ($)", self.cap_var, 1, 1, "e.g. 0")

        # ── Categorical card ──
        self._section("CATEGORICAL FEATURES")
        cat_card = self._card()
        g2 = tk.Frame(cat_card, bg=CARD_BG)
        g2.pack(fill="x", padx=14, pady=10)
        g2.columnconfigure(0, weight=1, uniform="c")
        g2.columnconfigure(1, weight=1, uniform="c")

        self.wc_var = tk.StringVar(value=WORKCLASSES[0])
        self.ed_var = tk.StringVar(value=EDUCATIONS[0])
        self.ms_var = tk.StringVar(value=MARITAL_STATUS[2])
        self.oc_var = tk.StringVar(value=OCCUPATIONS[0])
        self.re_var = tk.StringVar(value=RELATIONSHIPS[3])
        self.ra_var = tk.StringVar(value=RACES[0])
        self.sx_var = tk.StringVar(value="Male")
        self.co_var = tk.StringVar(value=COUNTRIES[0])

        self._dropdown(g2, "Workclass",       self.wc_var, WORKCLASSES,    0, 0)
        self._dropdown(g2, "Education level", self.ed_var, EDUCATIONS,     0, 1)
        self._dropdown(g2, "Marital status",  self.ms_var, MARITAL_STATUS, 1, 0)
        self._dropdown(g2, "Occupation",      self.oc_var, OCCUPATIONS,    1, 1)
        self._dropdown(g2, "Relationship",    self.re_var, RELATIONSHIPS,  2, 0)
        self._dropdown(g2, "Race",            self.ra_var, RACES,          2, 1)
        self._dropdown(g2, "Sex",             self.sx_var, ["Male","Female"], 3, 0)
        self._dropdown(g2, "Native country",  self.co_var, COUNTRIES,      3, 1)

        # ── Buttons ──
        btn_row = tk.Frame(self.main, bg=BG)
        btn_row.pack(fill="x", padx=20, pady=(14, 6))

        self.predict_btn = tk.Button(
            btn_row, text="Predict Income",
            font=("Helvetica", 12, "bold"),
            bg=PURPLE, fg="white", relief="flat",
            activebackground="#6760c4", activeforeground="white",
            cursor="hand2", padx=20, pady=10,
            command=self._predict, state="disabled")
        self.predict_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        tk.Button(btn_row, text="Reset",
                  font=("Helvetica", 12), bg="#EEEEEE", fg=TEXT_MUTED,
                  relief="flat", activebackground="#DDDDDD",
                  cursor="hand2", padx=20, pady=10,
                  command=self._reset).pack(side="left")

        # Result area
        self.result_frame = tk.Frame(self.main, bg=BG)
        self.result_frame.pack(fill="x", padx=20, pady=(8, 24))

    # ── Widget helpers ───────────────────────────────────────────────────────
    def _section(self, text):
        tk.Label(self.main, text=text, font=("Helvetica", 9, "bold"),
                 bg=BG, fg=TEXT_MUTED).pack(anchor="w", padx=20, pady=(10, 2))

    def _card(self):
        f = tk.Frame(self.main, bg=CARD_BG,
                     highlightbackground=BORDER, highlightthickness=1)
        f.pack(fill="x", padx=20, pady=(0, 4))
        return f

    def _textbox(self, parent, label, var, row, col, hint=""):
        f = tk.Frame(parent, bg=CARD_BG)
        f.grid(row=row, column=col, sticky="ew", padx=8, pady=8)
        tk.Label(f, text=label, font=("Helvetica", 10),
                 bg=CARD_BG, fg=TEXT_LABEL).pack(anchor="w")
        tk.Entry(f, textvariable=var, font=("Helvetica", 12),
                 relief="flat", bg="#F4F4F4", fg=TEXT_MAIN,
                 insertbackground=TEXT_MAIN,
                 highlightbackground=BORDER, highlightthickness=1,
                 width=16).pack(fill="x", pady=(3, 2), ipady=6)
        if hint:
            tk.Label(f, text=hint, font=("Helvetica", 9),
                     bg=CARD_BG, fg="#AAAAAA").pack(anchor="w")

    def _dropdown(self, parent, label, var, options, row, col):
        f = tk.Frame(parent, bg=CARD_BG)
        f.grid(row=row, column=col, sticky="ew", padx=8, pady=8)
        tk.Label(f, text=label, font=("Helvetica", 10),
                 bg=CARD_BG, fg=TEXT_LABEL).pack(anchor="w")
        ttk.Combobox(f, textvariable=var, values=options,
                     state="readonly", font=("Helvetica", 11),
                     width=22).pack(fill="x", pady=(3, 0))

    # ── Predict ──────────────────────────────────────────────────────────────
    def _predict(self):
        if not self.model_obj.trained:
            messagebox.showinfo("Please wait", "Model is still loading, please wait a moment.")
            return

        errs = []
        def get_int(var, name, lo, hi):
            try:
                v = int(var.get())
                if not (lo <= v <= hi):
                    raise ValueError
                return v
            except ValueError:
                errs.append(f"• {name}: enter a number between {lo} and {hi}")
                return None

        age = get_int(self.age_var, "Age",             17, 90)
        hrs = get_int(self.hrs_var, "Hours per week",   1, 99)
        edu = get_int(self.edu_var, "Education years",  1, 16)
        cap = get_int(self.cap_var, "Capital gain",     0, 999999)

        if errs:
            messagebox.showerror("Invalid input", "\n".join(errs))
            return

        row = {
            "age":            age,
            "workclass":      self.wc_var.get(),
            "education":      self.ed_var.get(),
            "education-num":  edu,
            "marital-status": self.ms_var.get(),
            "occupation":     self.oc_var.get(),
            "relationship":   self.re_var.get(),
            "race":           self.ra_var.get(),
            "sex":            1 if self.sx_var.get() == "Male" else 0,
            "capital-gain":   cap,
            "capital-loss":   0,
            "hours-per-week": hrs,
            "native-country": self.co_var.get(),
        }

        try:
            pred, proba = self.model_obj.predict(row)
            self._show_result(pred, proba)
        except Exception as ex:
            messagebox.showerror("Prediction error", str(ex))

    def _show_result(self, pred, proba):
        for w in self.result_frame.winfo_children():
            w.destroy()

        label   = ">50K"  if pred == 1 else "≤50K"
        color   = GREEN   if pred == 1 else RED
        pct     = round(proba * 100)
        conf    = "High" if pct > 70 or pct < 30 else "Medium" if pct > 55 or pct < 45 else "Low"
        conf_bg = {"High": "#E8F5E9", "Medium": "#FFF8E1", "Low": "#FFEBEE"}[conf]
        conf_fg = {"High": GREEN,     "Medium": AMBER,     "Low": RED}[conf]

        card = tk.Frame(self.result_frame, bg=CARD_BG,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x")

        top = tk.Frame(card, bg=CARD_BG)
        top.pack(fill="x", padx=16, pady=(14, 8))

        left = tk.Frame(top, bg=CARD_BG)
        left.pack(side="left")
        tk.Label(left, text="prediction", font=("Helvetica", 10),
                 bg=CARD_BG, fg=TEXT_MUTED).pack(anchor="w")
        tk.Label(left, text=label, font=("Helvetica", 30, "bold"),
                 bg=CARD_BG, fg=color).pack(anchor="w")

        right = tk.Frame(top, bg=CARD_BG)
        right.pack(side="right")
        tk.Label(right, text="probability", font=("Helvetica", 10),
                 bg=CARD_BG, fg=TEXT_MUTED).pack(anchor="e")
        tk.Label(right, text=f"{pct}%", font=("Helvetica", 30, "bold"),
                 bg=CARD_BG, fg=TEXT_MAIN).pack(anchor="e")

        # Progress bar
        bar_bg = tk.Canvas(card, height=10, bg="#EEEEEE",
                           highlightthickness=0, relief="flat")
        bar_bg.pack(fill="x", padx=16, pady=(0, 10))
        bar_bg.update_idletasks()
        w = bar_bg.winfo_width()
        bar_bg.create_rectangle(0, 0, int(w * proba), 10, fill=color, outline="")

        # Confidence badge
        bot = tk.Frame(card, bg=CARD_BG)
        bot.pack(fill="x", padx=16, pady=(0, 14))
        tk.Label(bot, text=f"  {conf} confidence  ",
                 font=("Helvetica", 10), bg=conf_bg, fg=conf_fg,
                 padx=4, pady=3).pack(side="left")

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset(self):
        for var in [self.age_var, self.hrs_var, self.edu_var, self.cap_var]:
            var.set("")
        self.wc_var.set(WORKCLASSES[0])
        self.ed_var.set(EDUCATIONS[0])
        self.ms_var.set(MARITAL_STATUS[2])
        self.oc_var.set(OCCUPATIONS[0])
        self.re_var.set(RELATIONSHIPS[3])
        self.ra_var.set(RACES[0])
        self.sx_var.set("Male")
        self.co_var.set(COUNTRIES[0])
        for w in self.result_frame.winfo_children():
            w.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()