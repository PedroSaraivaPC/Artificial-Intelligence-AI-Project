import os
import random

# pasta onde tens as imagens
folder = r"C:\Users\Pedro\Desktop\ISEC\3o_Ano\IC\TrabalhoPratico\Meta1\dataset\train\SUV"

# lista com todos os ficheiros da pasta
files = [f for f in os.listdir(folder) if f.endswith(".png") or f.endswith(".jpg")]

# baralhar a lista
random.shuffle(files)

# renomear os ficheiros na nova ordem
for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]
    new_name = f"{i:000d}{ext}"
    os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))

print("Renomeação concluída!")
