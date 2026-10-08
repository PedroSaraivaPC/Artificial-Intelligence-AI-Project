# optimize_ackley_sensitivity.py

import numpy as np
import time
from SwarmPackagePy import gwo, pso

# definição da Função Benchmark (Ackley)
# O objetivo de qualquer otimizador é encontrar os valores de 'x' que resultem no menor 'y' (fitness) possível.
# Para a função Ackley, o melhor 'y' é 0.0, que acontece quando 'x' é [0, 0, ..., 0].

def ackley_function(x): #recebe um vetor 
    d = len(x)  # Dimensão
    a = 20  #constantes padrao para a funcao Ackley
    b = 0.2
    c = 2 * np.pi

    # converte a lista 'x' para um array numpy para cálculos eficientes
    x_arr = np.array(x)

    sum1 = np.sum(np.square(x_arr)) #vai meter todos os numeros do array em quadrado e somar
    sum2 = np.sum(np.cos(c * x_arr)) #cria as ondas/buracos para enganar o algoritmo

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d)) #formula de ackley 
    term2 = -np.exp(sum2 / d)

    # O valor a minimizar (fitness), o objetivo e 0.0
    return term1 + term2 + a + np.exp(1)


# grelhas de parâmetros para análise de sensibilidade


dims = [2, 3] # 2D e 3D, como pedido no enunciado

# conjuntos de parâmetros para o GWO (variação de população e iterações)
gwo_param_grid = [
    {"pop": 20, "iter": 50},
    {"pop": 50, "iter": 50},
    {"pop": 50, "iter": 100},
]

# conjuntos de parâmetros para o PSO (pop, iter + w/c1/c2 diferentes)
pso_param_grid = [
    {"pop": 20, "iter": 50, "w": 0.4, "c1": 1.5, "c2": 1.5},
    {"pop": 50, "iter": 50, "w": 0.7, "c1": 1.5, "c2": 1.5},
    {"pop": 50, "iter": 100, "w": 0.5, "c1": 2.0, "c2": 2.0},
]

# Número de repetições por combinação (se quiseres mais robustez podes aumentar)
repeats = 10  # podes mudar para 3 se o tempo de execução não for problema


# função auxiliar para correr uma experiência e medir tempo
def run_gwo(dim, pop_size, iterations):
    # Limites típicos da função Ackley em cada dimensão
    lb = [-32.768] * dim #canto inferior esquerdo do mapa , esse valor ja é oficial da funcao ackley
    ub = [32.768] * dim #canto superior direito

    start = time.time()#liga cronometro
    
    algo = gwo(pop_size, ackley_function, lb, ub, dim, iterations) #chama o gwo com aqueles parametros para resolver o problema
    end = time.time()#desliga cronometro

    # Melhor posição encontrada
    best_solution = np.array(algo.get_Gbest()) #converte a melhor solucao que o gwo encontrou em um array de np 
    
    # Melhor fitness correspondente
    best_fitness = float(ackley_function(best_solution)) #calcula o fitness da melhor solucao

    elapsed = end - start #tempo que demorou
    return best_solution, best_fitness, elapsed


def run_pso(dim, pop_size, iterations, w, c1, c2):
    # Limites típicos da função Ackley em cada dimensão
    lb = [-32.768] * dim
    ub = [32.768] * dim

    start = time.time()
    algo = pso(pop_size, ackley_function, lb, ub, dim, iterations, w, c1, c2)
    end = time.time()

    best_solution = np.array(algo.get_Gbest())
    best_fitness = float(ackley_function(best_solution))

    elapsed = end - start #tempo que demorou
    return best_solution, best_fitness, elapsed



# experiências + recolha de resultados

resultados = []  # lista de dicionários, depois imprimimos em formato tabela

print("--- OTIMIZAÇÃO DA FUNÇÃO DE ACKLEY (ANÁLISE DE SENSIBILIDADE) ---\n")
print("Ótimo global conhecido: Y* = 0.0 em X = [0, ..., 0]\n")

for dim in dims: #executa 2 vezes [2D,3D]

    print("=" * 50)
    print(f"Dimensão da função Ackley: d = {dim}")
    print("=" * 50)

    # --- GWO ---
    for cfg in gwo_param_grid: #executa 3 vezes para cada dimensao ou seja 6 vezes
        pop = cfg["pop"]
        it = cfg["iter"]
        
        best_fits = []
        times = []

        print(f"\n[GWO] dim={dim}, pop={pop}, iter={it} (repeats={repeats})")

        for r in range(repeats): #executa 10 vezes
            sol, fit, tsec = run_gwo(dim, pop, it)
            print(f"  Run {r+1}: fitness={fit:.6f}, tempo={tsec:.4f}s, X={np.round(sol, 5)}")
            best_fits.append(fit) #adiciona o fitness a lista da run
            times.append(tsec)

        # média (se repeats=1, é simplesmente o valor único)
        mean_fit = float(np.mean(best_fits))
        mean_time = float(np.mean(times))

        resultados.append({
            "alg": "GWO",
            "dim": dim,
            "pop": pop,
            "iter": it,
            "w": None,
            "c1": None,
            "c2": None,
            "best_fitness_mean": mean_fit,
            "time_mean": mean_time,
        })

    # --- PSO ---
    for cfg in pso_param_grid:
        pop = cfg["pop"]
        it = cfg["iter"]
        w = cfg["w"]
        c1 = cfg["c1"]
        c2 = cfg["c2"]

        best_fits = []
        times = []

        print(f"\n[PSO] dim={dim}, pop={pop}, iter={it}, w={w}, c1={c1}, c2={c2} (repeats={repeats})")

        for r in range(repeats):
            sol, fit, tsec = run_pso(dim, pop, it, w, c1, c2)
            print(f"  Run {r+1}: fitness={fit:.6f}, tempo={tsec:.4f}s, X={np.round(sol, 5)}")
            best_fits.append(fit)
            times.append(tsec)

        mean_fit = float(np.mean(best_fits))
        mean_time = float(np.mean(times))

        resultados.append({
            "alg": "PSO",
            "dim": dim,
            "pop": pop,
            "iter": it,
            "w": w,
            "c1": c1,
            "c2": c2,
            "best_fitness_mean": mean_fit,
            "time_mean": mean_time,
        })

# impressão em formato de tabela

print("\n\n" + "#" * 50)
print("# TABELA DE RESULTADOS - ANÁLISE DE SENSIBILIDADE (ACKLEY)")
print("# Cada linha corresponde à média das runs (repeats =", repeats, ")")
print("#" * 50 + "\n")

# Cabeçalho
header = [
    "Algoritmo",
    "Dim",
    "População",
    "Iterações",
    "w",
    "c1",
    "c2",
    "Melhor Fitness (médio)",
    "Tempo médio (s)"
]
print(";".join(header))

for r in resultados:
    row = [
        r["alg"],
        str(r["dim"]),
        str(r["pop"]),
        str(r["iter"]),
        "-" if r["w"] is None else f"{r['w']:.3f}", #isto e um ternario , se R for vazio mete - 
        "-" if r["c1"] is None else f"{r['c1']:.3f}",
        "-" if r["c2"] is None else f"{r['c2']:.3f}",
        f"{r['best_fitness_mean']:.6f}", #mete o fitness com 6 casas decimais
        f"{r['time_mean']:.4f}",
    ]
    print(";".join(row))

print("\nFim da otimização da função de Ackley.")
