import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ── LOAD FILES ────────────────────────────────────────────────────────────────
with open("best_model_name.txt")  as f: best_name = f.read().strip()
with open("feature_names.pkl",  "rb") as f: feature_names = pickle.load(f)
with open("scaler.pkl",         "rb") as f: scaler  = pickle.load(f)
with open("imputer.pkl",        "rb") as f: imputer = pickle.load(f)

model_map = {
    "Logistic Regression": "lr_model.pkl",
    "KNN":                 "knn_model.pkl",
    "Decision Tree":       "dt_model.pkl",
}
models = {name: pickle.load(open(path, "rb")) for name, path in model_map.items()}
best_model = models[best_name]

X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results = []
for name, mdl in models.items():
    preds = mdl.predict(X_test)
    results.append({
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test,  preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test,    preds), 4),
        "F1-Score":  round(f1_score(y_test,        preds), 4),
    })

# ── PALETTE ───────────────────────────────────────────────────────────────────
BG       = "#0A0E17"
CARD     = "#111827"
CARD2    = "#1A2235"
BORDER   = "#1E2D45"
ACCENT   = "#00D4FF"
ACCENT2  = "#0096B4"
SUCCESS  = "#00E676"
DANGER   = "#FF1744"
GOLD     = "#FFD600"
TXT      = "#E8F0FE"
SUBTEXT  = "#607D8B"
C_LR     = "#00D4FF"
C_KNN    = "#FF6B35"
C_DT     = "#A8FF78"

F_HERO   = ("Segoe UI", 22, "bold")
F_TITLE  = ("Segoe UI", 13, "bold")
F_LBL    = ("Segoe UI", 10)
F_SMALL  = ("Segoe UI", 9)
F_MONO   = ("Consolas", 10)
F_BTN    = ("Segoe UI", 11, "bold")
F_RESULT = ("Segoe UI", 17, "bold")

MODEL_COLORS = {"Logistic Regression": C_LR, "KNN": C_KNN, "Decision Tree": C_DT}

FEATURE_LABELS = {
    "age":      "Age",
    "sex":      "Sex  (0=Female · 1=Male)",
    "cp":       "Chest Pain Type  (0–3)",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Cholesterol",
    "fbs":      "Fasting Blood Sugar >120  (0/1)",
    "restecg":  "Resting ECG  (0–2)",
    "thalch":   "Max Heart Rate",
    "exang":    "Exercise Angina  (0/1)",
    "oldpeak":  "ST Depression",
    "slope":    "Slope of ST  (0–2)",
    "ca":       "Major Vessels  (0–4)",
    "thal":     "Thal  (0·1·2)",
}
DEFAULTS = {
    "age":"45","sex":"1","cp":"0","trestbps":"120","chol":"200",
    "fbs":"0","restecg":"0","thalch":"150","exang":"0",
    "oldpeak":"1.0","slope":"1","ca":"0","thal":"0",
}

# ── ROOT ──────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Heart Disease Risk Predictor — ML Dashboard")
root.geometry("1000x740")
root.configure(bg=BG)
root.resizable(True, True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
sidebar = tk.Frame(root, bg=CARD, width=200)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

tk.Frame(sidebar, bg=ACCENT, height=4).pack(fill="x")

tk.Label(sidebar, text="❤", bg=CARD, fg=DANGER,
         font=("Segoe UI", 28)).pack(pady=(24, 0))
tk.Label(sidebar, text="Heart Risk\nPredictor", bg=CARD, fg=TXT,
         font=("Segoe UI", 13, "bold"), justify="center").pack(pady=(4, 2))
tk.Label(sidebar, text="ML Dashboard", bg=CARD, fg=SUBTEXT,
         font=F_SMALL).pack(pady=(0, 24))

tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=16)

# Nav buttons
current_tab = tk.StringVar(value="predict")
nav_buttons = {}

def switch_tab(name):
    current_tab.set(name)
    for n, btn in nav_buttons.items():
        if n == name:
            btn.config(bg=ACCENT2, fg=BG)
        else:
            btn.config(bg=CARD, fg=SUBTEXT)
    for frame_name, frame in tab_frames.items():
        if frame_name == name:
            frame.pack(fill="both", expand=True)
        else:
            frame.pack_forget()

NAV = [
    ("predict",    "▶  Predict"),
    ("comparison", "📊  Comparison"),
    ("matrices",   "🔲  Confusion"),
    ("winner",     "🏆  Winner"),
]

tk.Frame(sidebar, bg=CARD, height=8).pack()
for key, label in NAV:
    btn = tk.Button(sidebar, text=label, bg=CARD, fg=SUBTEXT,
                    font=("Segoe UI", 10, "bold"),
                    relief="flat", anchor="w", padx=20, pady=10,
                    activebackground=ACCENT2, activeforeground=BG,
                    cursor="hand2",
                    command=lambda k=key: switch_tab(k))
    btn.pack(fill="x")
    nav_buttons[key] = btn

# Best model badge in sidebar
tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=16, pady=20)
tk.Label(sidebar, text="Best Model", bg=CARD, fg=SUBTEXT, font=F_SMALL).pack()
tk.Label(sidebar, text=best_name, bg=CARD, fg=GOLD,
         font=("Segoe UI", 10, "bold"), wraplength=160, justify="center").pack(pady=4)

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
main = tk.Frame(root, bg=BG)
main.pack(side="left", fill="both", expand=True)

tab_frames = {}

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
predict_frame = tk.Frame(main, bg=BG)
tab_frames["predict"] = predict_frame

# Header
ph = tk.Frame(predict_frame, bg=BG)
ph.pack(fill="x", padx=28, pady=(20, 8))
tk.Label(ph, text="Patient Data", bg=BG, fg=TXT, font=F_HERO).pack(anchor="w")
tk.Label(ph, text=f"Enter values below and click Predict  ·  Active model: {best_name}",
         bg=BG, fg=SUBTEXT, font=F_SMALL).pack(anchor="w")
tk.Frame(predict_frame, bg=BORDER, height=1).pack(fill="x", padx=28, pady=(0, 10))

# Scrollable form
wrap = tk.Frame(predict_frame, bg=BG)
wrap.pack(fill="both", expand=True, padx=28)

canvas_p = tk.Canvas(wrap, bg=BG, highlightthickness=0)
sb_p = ttk.Scrollbar(wrap, orient="vertical", command=canvas_p.yview)
canvas_p.configure(yscrollcommand=sb_p.set)
sb_p.pack(side="right", fill="y")
canvas_p.pack(fill="both", expand=True)

form = tk.Frame(canvas_p, bg=BG)
canvas_p.create_window((0, 0), window=form, anchor="nw")

entries = {}
for i, feat in enumerate(feature_names):
    lbl = FEATURE_LABELS.get(feat, feat)
    default = DEFAULTS.get(feat, "0")
    row = tk.Frame(form, bg=CARD if i % 2 == 0 else BG)
    row.pack(fill="x", pady=2, ipady=4)
    tk.Label(row, text=lbl, bg=row["bg"], fg=TXT,
             font=F_LBL, width=32, anchor="w", padx=8).pack(side="left")
    e = tk.Entry(row, bg=CARD2, fg=ACCENT, insertbackground=ACCENT,
                 font=F_MONO, width=12, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT)
    e.insert(0, default)
    e.pack(side="left", padx=8, ipady=3)
    entries[feat] = e

form.update_idletasks()
canvas_p.config(scrollregion=canvas_p.bbox("all"))
canvas_p.bind_all("<MouseWheel>",
    lambda e: canvas_p.yview_scroll(-1*(e.delta//120), "units"))

# Result + Button
tk.Frame(predict_frame, bg=BORDER, height=1).pack(fill="x", padx=28, pady=(8, 0))
result_var = tk.StringVar(value="")
prob_var   = tk.StringVar(value="")

result_lbl = tk.Label(predict_frame, textvariable=result_var,
                      bg=BG, font=F_RESULT, pady=4)
result_lbl.pack()
tk.Label(predict_frame, textvariable=prob_var,
         bg=BG, fg=SUBTEXT, font=F_SMALL).pack()

def predict():
    try:
        vals = [float(entries[f].get()) for f in feature_names]
        x = np.array(vals).reshape(1, -1)
        x = imputer.transform(x)
        x = scaler.transform(x)
        pred  = best_model.predict(x)[0]
        proba = best_model.predict_proba(x)[0]
        if pred == 1:
            result_var.set("⚠   HIGH RISK — Heart Disease Likely")
            result_lbl.config(fg=DANGER)
            prob_var.set(f"Confidence: {proba[1]*100:.1f}%")
        else:
            result_var.set("✔   LOW RISK — No Heart Disease")
            result_lbl.config(fg=SUCCESS)
            prob_var.set(f"Confidence: {proba[0]*100:.1f}%")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

tk.Button(predict_frame, text="▶   RUN PREDICTION",
          bg=ACCENT, fg=BG, font=F_BTN, relief="flat",
          padx=30, pady=10, cursor="hand2",
          activebackground="#79E0FF",
          command=predict).pack(pady=(4, 16))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
comp_frame = tk.Frame(main, bg=BG)
tab_frames["comparison"] = comp_frame

tk.Label(comp_frame, text="Model Comparison", bg=BG, fg=TXT,
         font=F_HERO).pack(anchor="w", padx=28, pady=(20, 2))
tk.Label(comp_frame, text="Accuracy · Precision · Recall · F1-Score across all models",
         bg=BG, fg=SUBTEXT, font=F_SMALL).pack(anchor="w", padx=28)
tk.Frame(comp_frame, bg=BORDER, height=1).pack(fill="x", padx=28, pady=(8, 0))

# Metrics table
table_frame = tk.Frame(comp_frame, bg=CARD, bd=0)
table_frame.pack(fill="x", padx=28, pady=12)

headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
col_w   = [22, 10, 10, 10, 10]
for j, h in enumerate(headers):
    tk.Label(table_frame, text=h, bg=CARD2, fg=ACCENT,
             font=("Segoe UI", 10, "bold"),
             width=col_w[j], pady=6, relief="flat").grid(row=0, column=j, padx=1, pady=1)

best_f1 = max(r["F1-Score"] for r in results)
for i, row in enumerate(results):
    is_best = row["F1-Score"] == best_f1
    color   = MODEL_COLORS[row["Model"]]
    for j, key in enumerate(headers):
        val  = row[key] if key != "Model" else row["Model"]
        fg   = GOLD if (key == "F1-Score" and is_best) else TXT
        font = ("Segoe UI", 10, "bold") if is_best else ("Segoe UI", 10)
        tk.Label(table_frame, text=str(val), bg=CARD,
                 fg=fg, font=font,
                 width=col_w[j], pady=5).grid(row=i+1, column=j, padx=1, pady=1)

# Bar chart
fig_c, ax_c = plt.subplots(figsize=(7, 3.2))
fig_c.patch.set_facecolor(BG)
ax_c.set_facecolor(CARD)

metrics  = ["Accuracy", "Precision", "Recall", "F1-Score"]
x        = np.arange(len(metrics))
width    = 0.25
bar_cols = [C_LR, C_KNN, C_DT]

for i, row in enumerate(results):
    vals = [row[m] for m in metrics]
    bars = ax_c.bar(x + i*width, vals, width,
                    label=row["Model"], color=bar_cols[i],
                    alpha=0.9, zorder=3)
    for bar in bars:
        ax_c.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 0.008,
                  f"{bar.get_height():.2f}",
                  ha="center", va="bottom",
                  fontsize=7.5, color=TXT, zorder=4)

ax_c.set_xticks(x + width)
ax_c.set_xticklabels(metrics, color=TXT, fontsize=10)
ax_c.set_ylim(0, 1.12)
ax_c.set_ylabel("Score", color=SUBTEXT, fontsize=9)
ax_c.tick_params(colors=TXT)
ax_c.legend(facecolor=CARD2, labelcolor=TXT, fontsize=9,
            edgecolor=BORDER)
ax_c.grid(axis="y", linestyle="--", alpha=0.2, color=TXT, zorder=0)
for spine in ax_c.spines.values():
    spine.set_edgecolor(BORDER)
fig_c.tight_layout()

canvas_c = FigureCanvasTkAgg(fig_c, master=comp_frame)
canvas_c.draw()
canvas_c.get_tk_widget().pack(fill="both", expand=True, padx=28, pady=(0, 16))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
mat_frame = tk.Frame(main, bg=BG)
tab_frames["matrices"] = mat_frame

tk.Label(mat_frame, text="Confusion Matrices", bg=BG, fg=TXT,
         font=F_HERO).pack(anchor="w", padx=28, pady=(20, 2))
tk.Label(mat_frame, text="True vs Predicted labels for each model",
         bg=BG, fg=SUBTEXT, font=F_SMALL).pack(anchor="w", padx=28)
tk.Frame(mat_frame, bg=BORDER, height=1).pack(fill="x", padx=28, pady=(8, 4))

fig_m, axes_m = plt.subplots(1, 3, figsize=(9, 3.4))
fig_m.patch.set_facecolor(BG)

cmaps = ["Blues", "Oranges", "Greens"]
for idx, (name, mdl) in enumerate(models.items()):
    ax  = axes_m[idx]
    cm  = confusion_matrix(y_test, mdl.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[idx], ax=ax,
                xticklabels=["Low", "High"],
                yticklabels=["Low", "High"],
                linewidths=0.5, linecolor=BG,
                cbar=False)
    ax.set_facecolor(CARD)
    ax.set_title(name, color=MODEL_COLORS[name],
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted", color=SUBTEXT, fontsize=8)
    ax.set_ylabel("Actual",    color=SUBTEXT, fontsize=8)
    ax.tick_params(colors=TXT)

fig_m.suptitle("", fontsize=1)
fig_m.tight_layout(pad=1.5)

canvas_m = FigureCanvasTkAgg(fig_m, master=mat_frame)
canvas_m.draw()
canvas_m.get_tk_widget().pack(fill="both", expand=True, padx=28, pady=(0, 16))

# Per-model accuracy under matrices
stat_row = tk.Frame(mat_frame, bg=BG)
stat_row.pack(fill="x", padx=28, pady=(0, 12))
for row in results:
    col = MODEL_COLORS[row["Model"]]
    card = tk.Frame(stat_row, bg=CARD, bd=0)
    card.pack(side="left", expand=True, fill="x", padx=6, pady=4, ipady=8)
    tk.Label(card, text=row["Model"], bg=CARD, fg=col,
             font=("Segoe UI", 9, "bold")).pack()
    tk.Label(card, text=f"Acc: {row['Accuracy']*100:.1f}%",
             bg=CARD, fg=TXT, font=F_MONO).pack()
    tk.Label(card, text=f"F1:  {row['F1-Score']*100:.1f}%",
             bg=CARD, fg=TXT, font=F_MONO).pack()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WINNER
# ═══════════════════════════════════════════════════════════════════════════════
win_frame = tk.Frame(main, bg=BG)
tab_frames["winner"] = win_frame

tk.Label(win_frame, text="🏆  Winner", bg=BG, fg=TXT,
         font=F_HERO).pack(anchor="w", padx=28, pady=(20, 2))
tk.Label(win_frame, text="Best performing model based on F1-Score",
         bg=BG, fg=SUBTEXT, font=F_SMALL).pack(anchor="w", padx=28)
tk.Frame(win_frame, bg=BORDER, height=1).pack(fill="x", padx=28, pady=(8, 20))

best_row = next(r for r in results if r["Model"] == best_name)
win_col  = MODEL_COLORS[best_name]

# Trophy card
trophy = tk.Frame(win_frame, bg=CARD, bd=0)
trophy.pack(padx=60, pady=8, fill="x")
tk.Frame(trophy, bg=win_col, height=5).pack(fill="x")

tk.Label(trophy, text="🥇", bg=CARD,
         font=("Segoe UI", 48)).pack(pady=(18, 4))
tk.Label(trophy, text=best_name, bg=CARD, fg=GOLD,
         font=("Segoe UI", 22, "bold")).pack()
tk.Label(trophy, text="Best Model by F1-Score", bg=CARD,
         fg=SUBTEXT, font=F_SMALL).pack(pady=(2, 16))

# Stats inside trophy card
stats_inner = tk.Frame(trophy, bg=CARD)
stats_inner.pack(pady=(0, 20))
for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    val = best_row[metric]
    row_f = tk.Frame(stats_inner, bg=CARD)
    row_f.pack(pady=3)
    color = GOLD if metric == "F1-Score" else TXT
    tk.Label(row_f, text=f"{metric}:", bg=CARD, fg=SUBTEXT,
             font=("Segoe UI", 11), width=12, anchor="e").pack(side="left")
    tk.Label(row_f, text=f"{val*100:.2f}%", bg=CARD, fg=color,
             font=("Segoe UI", 13, "bold"), width=10, anchor="w").pack(side="left")

# Ranking of all models
tk.Label(win_frame, text="Full Ranking", bg=BG, fg=TXT,
         font=F_TITLE).pack(anchor="w", padx=60, pady=(16, 6))

ranked = sorted(results, key=lambda r: r["F1-Score"], reverse=True)
medals = ["🥇", "🥈", "🥉"]
for idx, row in enumerate(ranked):
    col = MODEL_COLORS[row["Model"]]
    rank_card = tk.Frame(win_frame, bg=CARD)
    rank_card.pack(fill="x", padx=60, pady=4, ipady=6)
    tk.Label(rank_card, text=medals[idx], bg=CARD,
             font=("Segoe UI", 14), width=3).pack(side="left", padx=8)
    tk.Label(rank_card, text=row["Model"], bg=CARD, fg=col,
             font=("Segoe UI", 11, "bold"), width=22, anchor="w").pack(side="left")
    tk.Label(rank_card, text=f"F1: {row['F1-Score']*100:.2f}%",
             bg=CARD, fg=TXT, font=F_MONO).pack(side="left", padx=16)
    tk.Label(rank_card, text=f"Acc: {row['Accuracy']*100:.2f}%",
             bg=CARD, fg=SUBTEXT, font=F_MONO).pack(side="left")

# ── INIT ──────────────────────────────────────────────────────────────────────
switch_tab("predict")
root.mainloop()