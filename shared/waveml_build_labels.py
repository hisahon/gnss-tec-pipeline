# -*- coding: utf-8 -*-
"""
waveml_build_labels.py — transforma sua seleção manual de figuras em rótulos de treino.

Fica em: shared/waveml_build_labels.py

Como usar (mesmo fluxo que você já tem hoje):
  1. Rode WaveML_Assemble.py -> figuras em ml_model/figures_spectra/<rodada>/
  2. Olhe as figuras como sempre. As interessantes, MOVA (ou copie) para uma
     subpasta chamada 'selecionadas' dentro da pasta da rodada:
         ml_model/figures_spectra/2024333_1317/selecionadas/run_..._STAT-G05.png
  3. Rode, a partir da raiz do projeto:
         python shared/waveml_build_labels.py
     (ou, como módulo do pacote:  python -m shared.waveml_build_labels)
     -> gera ml_model/labels/labels.csv com:  sample_id, label (1=selecionada, 0=não)

IMPORTANTE — universo de negativos:
  O ponto de partida NÃO são as figuras, e sim as tabelas de features
  (ml_model/features/features_*.parquet). Todo par que passou pelo controle de
  qualidade (comprimento mínimo, sem buracos) entra como negativo (label 0)
  por padrão — mesmo que nunca tenha virado figura por não cruzar o
  threshold de amplitude. Isso é essencial em dias "de controle" (dias
  quietos, sem evento conhecido, que você roda de propósito só para dar
  ao modelo exemplos de calmaria completa): nesses dias pode não sair
  NENHUMA figura, mas as features continuam sendo a fonte de negativos.

  As figuras só entram para dar o sinal POSITIVO: qualquer PNG que você
  moveu para uma subpasta 'selecionadas/' vira label 1.

Regras:
  - Todo sample_id presente nas features -> label 0 por padrão
  - Todo sample_id cuja figura foi movida para 'selecionadas/' -> label 1
  - O prefixo de score (pNNN_) é removido automaticamente do nome, então
    você pode rodar de novo depois que o modelo já estiver ranqueando.
"""

import re
import csv
from pathlib import Path

import pandas as pd

FEATURES_DIR = Path("ml_model/features")
FIGURES_DIR = Path("ml_model/figures_spectra")
SELECTED_SUBDIR = "selecionadas"     # nome da subpasta onde você joga as boas
LABELS_DIR = Path("ml_model/labels")
LABELS_CSV = LABELS_DIR / "labels.csv"

SCORE_PREFIX = re.compile(r"^p\d{3}_")   # remove 'p087_' etc.


def sample_id_from_figure(path: Path) -> str:
    stem = path.stem
    return SCORE_PREFIX.sub("", stem)


def main():
    feature_files = sorted(FEATURES_DIR.glob("features_*.parquet"))
    if not feature_files:
        raise SystemExit(f"Nenhum parquet em {FEATURES_DIR}. Rode a rotina principal antes.")

    feats = pd.concat([pd.read_parquet(f) for f in feature_files], ignore_index=True)
    feats = feats.drop_duplicates(subset="sample_id", keep="last")

    # universo completo = todo par que passou no QC, negativo por padrão
    labels: dict[str, int] = {sid: 0 for sid in feats["sample_id"]}

    n_pos = 0
    unmatched = []
    if FIGURES_DIR.exists():
        for sel_dir in FIGURES_DIR.glob(f"*/{SELECTED_SUBDIR}"):
            for fig in sel_dir.glob("*.png"):
                sid = sample_id_from_figure(fig)
                if sid in labels:
                    labels[sid] = 1
                    n_pos += 1
                else:
                    # figura selecionada mas sem features correspondentes
                    # (rodada antiga, ou arquivo de features apagado)
                    unmatched.append(sid)

    if unmatched:
        print(f"[labels][aviso] {len(unmatched)} figura(s) selecionada(s) sem features "
              f"correspondentes — ignoradas. Ex.: {unmatched[:3]}")

    n_pos_final = sum(labels.values())
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "label"])
        for sid in sorted(labels):
            w.writerow([sid, labels[sid]])

    print(f"[labels] {len(labels)} exemplos únicos "
          f"({n_pos_final} positivos, {len(labels) - n_pos_final} negativos)")
    print(f"[labels] Salvo em {LABELS_CSV}")
    if n_pos_final < 20:
        print("[labels][aviso] Menos de 20 positivos — o modelo ainda vai ficar fraco. "
              "Continue selecionando manualmente por mais algumas rodadas.")


if __name__ == "__main__":
    main()
