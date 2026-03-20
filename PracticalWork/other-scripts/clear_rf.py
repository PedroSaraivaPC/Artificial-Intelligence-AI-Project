import os
import re
from pathlib import Path

EXTENSOES_IMAGENS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def normalizar_base(nome_sem_ext: str) -> str:
    """
    Remove espaços, hífens, underscores e outros símbolos.
    Fica só com letras e números, tudo em minúsculas.
    """
    nome_sem_ext = nome_sem_ext.lower()
    # remove tudo o que NÃO for letra ou número
    return re.sub(r"[^a-z0-9]+", "", nome_sem_ext)


def chave_para_ficheiro(f: Path) -> tuple[str, bool] | None:
    """
    Devolve (chave_normalizada, is_rf) para o ficheiro f,
    ou None se não for imagem.
    """
    ext = f.suffix.lower()
    if ext not in EXTENSOES_IMAGENS:
        return None

    nome = f.name
    nome_sem_ext = f.stem  # sem a última extensão (.jpg etc.)
    lower = nome.lower()

    if ".rf." in lower:
        # Ficheiro do tipo ..._jpg.rf.xxxxx.jpg
        idx_rf = lower.index(".rf.")
        # parte antes de ".rf."
        antes_rf = nome[:idx_rf]                # mantém maiúsculas/minúsculas originais
        # remover eventual extensão repetida no fim: "_jpg", "_jpeg", "_png"
        for suf in ("_jpg", "_jpeg", "_png"):
            if antes_rf.lower().endswith(suf):
                antes_rf = antes_rf[: -len(suf)]
                break
        chave = normalizar_base(antes_rf)
        return chave, True
    else:
        # Ficheiro normal (sem .rf.)
        chave = normalizar_base(nome_sem_ext)
        return chave, False


def apagar_bases_rf(pasta: Path):
    pasta = pasta.resolve()
    if not pasta.is_dir():
        print(f"Pasta não existe: {pasta}")
        return

    bases_por_chave = {}
    rf_por_chave = {}

    for f in pasta.iterdir():
        if not f.is_file():
            continue

        res = chave_para_ficheiro(f)
        if res is None:
            continue

        chave, is_rf = res

        if is_rf:
            rf_por_chave.setdefault(chave, []).append(f)
        else:
            bases_por_chave.setdefault(chave, []).append(f)

    a_apagar = []

    for chave, bases in bases_por_chave.items():
        if chave in rf_por_chave:
            # Há pelo menos um ficheiro RF com a mesma chave → apagar bases
            a_apagar.extend(bases)

    if not a_apagar:
        print("Nenhuma imagem 'base' correspondente a ficheiros .rf encontrada para apagar.")
        return

    print("Vai APAGAR estes ficheiros (vai manter os que têm .rf):")
    for f in sorted(a_apagar):
        print(" -", f.name)

    confirma = input("Confirmar? (s/N): ").strip().lower()
    if confirma == "s":
        for f in sorted(a_apagar):
            try:
                f.unlink()
                print("Apagado:", f.name)
            except Exception as e:
                print("Erro ao apagar", f.name, "->", e)
    else:
        print("Nada foi apagado.")


if __name__ == "__main__":
    caminho = input(
        "Caminho da pasta com as imagens (ENTER = pasta onde está este script): "
    ).strip()

    if caminho:
        pasta = Path(caminho)
    else:
        pasta = Path(__file__).parent  # pasta onde o script está

    apagar_bases_rf(pasta)
