"""
Telco müşteri terk tahmini - MLP sınıflandırma projesi.
Farklı deney senaryolarında çok katmanlı algılayıcı modellerini eğitir.
"""
import warnings
warnings.filterwarnings('ignore')

import time, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROJECT_INFO = {"Ders": "Yapay Sinir Ağlarına Giriş", "Öğrenci": "Ahmet Sümer - 221213028"}
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (48,), (64, 32), (48, 24)],
    "mlp__alpha": [0.0001, 0.001],
    "mlp__learning_rate_init": [0.001, 0.01],
}


@dataclass
class Result:
    name: str
    params: Dict[str, Any]
    accuracy: float
    confusion: np.ndarray


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def get_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    return X, y, num, cat


def make_pipeline(num_f: List[str], cat_f: List[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_f),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_f),
    ])
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42,
                        solver="adam", activation="relu", learning_rate="adaptive",
                        early_stopping=True, n_iter_no_change=20, verbose=False)
    return Pipeline([("preprocess", pre), ("mlp", mlp)])


def eval_train_test(X, y, pipe_fn, params) -> Result:
    plist = list(ParameterGrid(params))
    best = {"score": -np.inf, "params": None, "conf": None}
    
    for i, p in enumerate(plist):
        pipe = pipe_fn()
        pipe.set_params(**p)
        pipe.fit(X, y)
        acc = accuracy_score(y, pipe.predict(X))
        if acc > best["score"]:
            best = {"score": acc, "params": p, "conf": confusion_matrix(y, pipe.predict(X))}
        print(f"\r  Parametre arama: {i+1}/{len(plist)}", end="")
    print()
    
    return Result("Eğitim verisini test olarak kullanma", best["params"], best["score"], best["conf"])


def eval_cv(X, y, pipe_fn, params, folds) -> Result:
    print(f"  {folds}-Fold CV çalışıyor...")
    grid = GridSearchCV(pipe_fn(), params, cv=folds, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X, y)
    preds = cross_val_predict(grid.best_estimator_, X, y, cv=folds, n_jobs=-1)
    return Result(f"{folds}-fold cross validation", grid.best_params_, 
                  accuracy_score(y, preds), confusion_matrix(y, preds))


def eval_holdout(X, y, pipe_fn, params, runs=5) -> List[Result]:
    results = []
    plist = list(ParameterGrid(params))
    
    for i in range(runs):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42+i)
        best = {"score": -np.inf, "params": None, "conf": None}
        
        for p in plist:
            pipe = pipe_fn()
            pipe.set_params(**p)
            pipe.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, pipe.predict(X_te))
            if acc > best["score"]:
                best = {"score": acc, "params": p, "conf": confusion_matrix(y_te, pipe.predict(X_te))}
        
        print(f"\r  Holdout: Deneme {i+1}/{runs} tamamlandı", end="")
        results.append(Result(f"%75-%25 holdout - Deneme {i+1}", best["params"], best["score"], best["conf"]))
    print()
    return results


def plot_network(input_dim, hidden, output_dim, path, exp_name, acc):
    layers = [input_dim, *hidden, output_dim]
    labels = ["Girdi"] + [f"Gizli {i+1}" for i in range(len(hidden))] + ["Çıktı"]
    colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#95E1D3', '#A8E6CF']
    
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    
    n = len(layers)
    x_pos = np.linspace(1, 11, n)
    v_sp = min(0.8, 6.5 / max(layers))
    positions = []
    
    for li in range(n):
        n_neu = min(layers[li], 12)
        y_start = (7 - (n_neu-1)*v_sp) / 2 + 1.5
        positions.append([(x_pos[li], y_start + i*v_sp) for i in range(n_neu)])
    
    for li in range(1, n):
        for x1, y1 in positions[li-1]:
            for x2, y2 in positions[li]:
                ax.plot([x1, x2], [y1, y2], color='#DDA0DD', lw=0.8, alpha=0.3)
    
    for li in range(n):
        col = colors[0] if li == 0 else colors[-1] if li == n-1 else colors[1 + (li-1) % 3]
        for x, y in positions[li]:
            ax.add_patch(plt.Circle((x, y), 0.2, fc=col, ec='white', lw=2, zorder=3))
        ax.text(x_pos[li], 0.3, labels[li], ha='center', fontsize=10, fontweight='bold', color='#EAEAEA')
        ax.text(x_pos[li], -0.3, str(layers[li]), ha='center', fontsize=9, fontweight='bold', color='white')
    
    ax.text(6, 9.3, 'MLP Ağ Topolojisi', ha='center', fontsize=16, fontweight='bold', color='#EAEAEA')
    ax.text(6, 8.7, exp_name, ha='center', fontsize=12, color='#E94560', style='italic')
    
    badge_col = '#27ae60' if acc > 0.80 else '#f39c12' if acc > 0.70 else '#e74c3c'
    ax.add_patch(FancyBboxPatch((4.5, 7.9), 3, 0.6, boxstyle="round,pad=0.1", fc=badge_col, alpha=0.9))
    ax.text(6, 8.2, f'Doğruluk: {acc*100:.2f}%', ha='center', fontsize=11, fontweight='bold', color='white')
    
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.5, 10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)


def print_result(r: Result):
    hl = r.params.get("mlp__hidden_layer_sizes", "-")
    alpha = r.params.get("mlp__alpha", "-")
    lr = r.params.get("mlp__learning_rate_init", "-")
    
    print(f"\n  [{r.name}]")
    print(f"  Doğruluk: {r.accuracy*100:.2f}%")
    print(f"  Parametreler: Gizli={hl}, Alpha={alpha}, LR={lr}")
    print(f"  Konfüzyon: TN={r.confusion[0][0]}, FP={r.confusion[0][1]}, FN={r.confusion[1][0]}, TP={r.confusion[1][1]}")


def main():
    print("\n" + "="*50)
    print("  TELCO MÜŞTERİ TERK TAHMİNİ - MLP PROJESİ")
    print("="*50)
    
    print("\n[Proje Bilgileri]")
    for k, v in PROJECT_INFO.items():
        print(f"  {k}: {v}")
    
    print("\n[Veri Seti Yükleniyor]")
    df = load_data(DATA_PATH)
    X, y, num_f, cat_f = get_features(df)
    print(f"  Toplam: {len(df)} kayıt | Churn oranı: {y.mean():.2%}")
    
    pipe_fn = lambda: make_pipeline(num_f, cat_f)
    
    print("\n" + "="*50)
    print("  DENEYSEL ÇALIŞMALAR")
    print("="*50)
    
    print("\n[Deney 1: Eğitim = Test]")
    r1 = eval_train_test(X, y, pipe_fn, PARAM_GRID)
    print_result(r1)
    
    print("\n[Deney 2: 5-Fold CV]")
    r2 = eval_cv(X, y, pipe_fn, PARAM_GRID, 5)
    print_result(r2)
    
    print("\n[Deney 3: 10-Fold CV]")
    r3 = eval_cv(X, y, pipe_fn, PARAM_GRID, 10)
    print_result(r3)
    
    print("\n[Deney 4: %75-%25 Holdout]")
    holdouts = eval_holdout(X, y, pipe_fn, PARAM_GRID, 5)
    for r in holdouts:
        print_result(r)
    
    all_results = [r1, r2, r3, *holdouts]
    
    print("\n" + "="*50)
    print("  AĞ GÖRSELLEŞTİRMELERİ")
    print("="*50)
    
    input_dim = len(num_f) + len(cat_f)
    best_holdout = max(holdouts, key=lambda x: x.accuracy)
    
    experiments = [(r1, "1_Egitim_Verisi_Test"), (r2, "2_5fold_CrossValidation"),
                   (r3, "3_10fold_CrossValidation"), (best_holdout, "4_Holdout_75-25_EnBasarili")]
    
    for i, (r, suffix) in enumerate(experiments, 1):
        hl = r.params.get("mlp__hidden_layer_sizes", (32, 16))
        fname = f"telco_ag_deney{i}_{suffix}.png"
        plot_network(input_dim, hl, 1, Path(fname), r.name, r.accuracy)
        print(f"  {fname} oluşturuldu")
    
    print("\n" + "="*50)
    print("  SONUÇ")
    print("="*50)
    best = max(all_results, key=lambda x: x.accuracy)
    print(f"  En başarılı: {best.name}")
    print(f"  Doğruluk: {best.accuracy*100:.2f}%")
    
    print("\n  Tüm deneyler tamamlandı!")
    print("  GUI için: python telco_gui.py")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
