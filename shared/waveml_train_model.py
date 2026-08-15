# -*- coding: utf-8 -*-
"""
waveml_train_model.py — treina o classificador que aprende a sua seleção manual.

Fica em: shared/waveml_train_model.py

Pré-requisitos:
  - ml_model/features/features_*.parquet  (gerados pelo WaveML_Assemble.py)
  - ml_model/labels/labels.csv            (gerado pelo shared/waveml_build_labels.py)

Como rodar, a partir da raiz do projeto:
    python shared/waveml_train_model.py
(ou, como módulo do pacote:  python -m shared.waveml_train_model)

Saída:
  - ml_model/selector.joblib  -> carregado automaticamente pelo WaveML_Assemble.py

Modelo: HistGradientBoostingClassifier (sklearn). Motivos:
  - lida nativamente com NaN (features que falharam em alguns pares);
  - funciona bem com poucas centenas de exemplos;
  - não precisa de normalização das features.

A validação é feita por GRUPO (dia/rodada): o modelo é testado em dias que
nunca viu no treino, que é o cenário real de uso — evita a fuga de informação
de ter pares do mesmo evento no treino e no teste.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.inspection import permutation_importance

FEATURES_DIR = Path("ml_model/features")
LABELS_CSV = Path("ml_model/labels/labels.csv")
MODEL_PATH = Path("ml_model/selector.joblib")

# Colunas de metadados que NÃO entram como feature
META_COLS = {"sample_id", "pair", "station", "prn", "year", "doy",
             "t_beg", "t_end", "figure_path", "ml_score", "above_threshold"}


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    files = sorted(FEATURES_DIR.glob("features_*.parquet"))
    if not files:
        raise SystemExit(f"Nenhum parquet em {FEATURES_DIR}. Rode a rotina principal antes.")
    feats = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Se o mesmo sample_id foi reprocessado, mantém a versão mais recente
    feats = feats.drop_duplicates(subset="sample_id", keep="last")

    if not LABELS_CSV.exists():
        raise SystemExit(f"{LABELS_CSV} não encontrado. Rode shared/waveml_build_labels.py antes.")
    labels = pd.read_csv(LABELS_CSV)

    df = feats.merge(labels, on="sample_id", how="inner")
    print(f"[train] {len(feats)} amostras com features, {len(labels)} rotuladas, "
          f"{len(df)} casadas.")
    if len(df) < len(labels):
        print("[train][aviso] Alguns rótulos não casaram com features — "
              "confira se os parquets das rodadas correspondentes existem.")

    feature_names = [c for c in df.columns if c not in META_COLS and c != "label"]
    return df, feature_names


def main():
    df, feature_names = load_dataset()
    X = df[feature_names]
    y = df["label"].astype(int).to_numpy()
    groups = (df["year"].astype(str) + "_" + df["doy"].astype(str)).to_numpy()

    n_pos = int(y.sum())
    print(f"[train] {len(y)} exemplos | {n_pos} positivos ({100*n_pos/len(y):.1f}%) | "
          f"{len(np.unique(groups))} dias distintos | {len(feature_names)} features")
    if n_pos < 15:
        print("[train][aviso] Poucos positivos — resultados serão instáveis. "
              "Rotule mais algumas rodadas antes de confiar no modelo.")

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,          # árvores rasas: menos overfitting com poucos dados
        min_samples_leaf=10,
        l2_regularization=1.0,
        class_weight="balanced",    # positivos são raros
        random_state=42,
    )

    # ---------- Validação cruzada agrupada por dia ----------
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits)
        proba = cross_val_predict(model, X, y, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba)
        ap = average_precision_score(y, proba)
        print(f"\n[cv] ROC-AUC = {auc:.3f} | Average Precision = {ap:.3f} "
              f"({n_splits} folds, agrupado por dia)")

        # Precision@k: das k figuras mais bem ranqueadas, quantas você teria selecionado?
        order = np.argsort(-proba)
        for k in (10, 25, 50):
            if k <= len(y):
                print(f"[cv] Precision@{k}: {y[order[:k]].mean():.2f}")

        print("\n[cv] Relatório no ponto de corte 0.5:")
        print(classification_report(y, (proba >= 0.5).astype(int),
                                    target_names=["descartada", "selecionada"],
                                    zero_division=0))
    else:
        print("[cv][aviso] Só um dia rotulado — impossível validar de forma honesta. "
              "Métricas puladas; o modelo será treinado mesmo assim.")

    # ---------- Treino final com todos os dados ----------
    model.fit(X, y)

    # Importância por permutação (mais confiável que impurity para esse caso)
    try:
        imp = permutation_importance(model, X, y, n_repeats=10,
                                     random_state=42, scoring="roc_auc")
        ranking = sorted(zip(feature_names, imp.importances_mean),
                         key=lambda t: -t[1])
        print("\n[importance] Top features (queda de AUC ao embaralhar):")
        for name, val in ranking[:12]:
            print(f"   {name:28s} {val:+.4f}")
    except Exception as e:
        print(f"[importance][aviso] {e}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model,
                 "feature_names": feature_names,
                 "n_train": len(y)}, MODEL_PATH)
    print(f"\n[train] Modelo salvo em {MODEL_PATH} — a rotina principal passa a usá-lo "
          f"automaticamente na próxima execução.")


if __name__ == "__main__":
    main()
