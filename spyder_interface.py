# -*- coding: utf-8 -*-
"""
Script Lançador para o Spyder.
Define os parâmetros e chama o pipeline principal.
"""
import subprocess
import sys

# --- DEFINA SEUS PARÂMETROS AQUI ---
ANO = "2025"
DOY = "298"
REDE = "both"     # Opções: "rbmc", "ramsac", "both"
PULAR = ["1"]        # Exemplo: ["1"] para pular download, ou [] para rodar tudo
# ----------------------------------

def rodar():
    # Constrói a lista de argumentos para o terminal
    cmd = [
        sys.executable,  # Garante que usa o mesmo Python do Spyder
        "run_pipeline.py",
        "--year", ANO,
        "--doy", DOY,
        "--network", REDE
    ]
    
    if PULAR:
        cmd.append("--skip")
        cmd.extend(PULAR)

    print(f"\n>>> Iniciando Pipeline para {ANO}/{DOY} ({REDE})")
    print(f">>> Comando: {' '.join(cmd)}\n")
    
    # Executa o processo e mostra a saída em tempo real
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    rodar()# -*- coding: utf-8 -*-

