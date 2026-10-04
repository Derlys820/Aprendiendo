import os
import re
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_HTML = os.path.join(BASE, "htmls")

def extraer_enlaces(ruta_archivo):
    """Devuelve la lista de destinos (con duplicados) de un HTML."""
    with open(ruta_archivo, encoding="utf-8") as f:
        html = f.read()
    hrefs = re.findall(r'href\s*=\s*["\']([^"\'#]+)["\']', html, re.IGNORECASE)
    return [os.path.splitext(os.path.basename(h))[0] for h in hrefs]

def construir_diccionario(carpeta):
    """Escanea la carpeta y arma {pagina: [destinos]} sin recibir nada precalculado."""
    archivos = sorted(a for a in os.listdir(carpeta) if a.lower().endswith(".html"))
    nodos = {os.path.splitext(a)[0] for a in archivos}
    red = {}
    for a in archivos:
        nombre = os.path.splitext(a)[0]
        destinos = extraer_enlaces(os.path.join(carpeta, a))
        validos = [d for d in destinos if d in nodos]
        if len(validos) != len(destinos):
            print(f"  Aviso: {nombre} tiene enlaces a paginas inexistentes (ignorados)")
        red[nombre] = validos
    return red

def construir_matriz_google(red, d=0.85):
    nodos = sorted(red.keys())
    m = len(nodos)
    idx = {n: i for i, n in enumerate(nodos)}
    P = np.zeros((m, m))
    for nodo, enlaces in red.items():
        i = idx[nodo]
        if len(enlaces) == 0:
            P[i, :] = 1.0 / m                  # Dead End
        else:
            for destino in enlaces:            # duplicados suman peso
                P[i, idx[destino]] += 1.0 / len(enlaces)
    M = d * P + (1.0 - d) / m * np.ones((m, m))
    return nodos, P, M


def metodo_potencia(M, tol=1e-8, max_iter=500):
    m = M.shape[0]
    pi = np.ones(m) / m
    historial = [pi.copy()]
    for _ in range(max_iter):
        pi_sig = pi @ M
        historial.append(pi_sig.copy())
        if np.linalg.norm(pi_sig - pi, 1) < tol:
            pi = pi_sig
            break
        pi = pi_sig
    return pi, np.array(historial)


def monte_carlo_surfer(M, pasos=150000):
    m = M.shape[0]
    acum = np.cumsum(M, axis=1)
    estado = np.random.randint(0, m)
    visitas = np.zeros(m)
    aleatorios = np.random.random(pasos)
    for u in aleatorios:
        visitas[estado] += 1
        estado = min(int(np.searchsorted(acum[estado], u)), m - 1)
    return visitas / pasos


def grafica_convergencia(nodos, historial, titulo):
    plt.figure(figsize=(7, 4))
    n = min(30, len(historial))
    for k, nombre in enumerate(nodos):
        plt.plot(range(n), historial[:n, k], marker="o", ms=3, label=nombre)
    plt.xlabel("Iteracion n"); plt.ylabel("pi(n)")
    plt.title(f"Convergencia - {titulo}")
    plt.legend(fontsize=7, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"conv_{titulo}.png", dpi=150)


def grafica_comparativa(nodos, pi_num, pi_mc, titulo):
    x = np.arange(len(nodos)); w = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, pi_num, w, label="Metodo potencia")
    plt.bar(x + w / 2, pi_mc, w, label="Monte Carlo")
    plt.xticks(x, nodos); plt.ylabel("PageRank")
    plt.title(f"Monte Carlo vs numerico - {titulo}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"comp_{titulo}.png", dpi=150)


def grafica_sensibilidad(red, titulo):
    ds = np.linspace(0.10, 0.99, 12)
    iteraciones = []
    for d in ds:
        _, _, Md = construir_matriz_google(red, d)
        _, hist = metodo_potencia(Md, tol=1e-8, max_iter=5000)
        iteraciones.append(len(hist) - 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ds, iteraciones, marker="o")
    plt.xlabel("Factor de amortiguacion d"); plt.ylabel("Iteraciones hasta converger")
    plt.title(f"Sensibilidad a d - {titulo}")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"sens_{titulo}.png", dpi=150)


def grafica_ranking(nodos, pi, titulo):
    orden = np.argsort(pi)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar([nodos[i] for i in orden], pi[orden])
    plt.ylabel("PageRank"); plt.title(f"Ranking final - {titulo}")
    plt.tight_layout()
    plt.savefig(f"rank_{titulo}.png", dpi=150)


def imprimir_matriz(titulo, A, nodos):
    """Imprime una matriz con los nombres de los nodos en filas y columnas."""
    print(f"\n{titulo}")
    print(" " * 6 + "".join(f"{n:>8}" for n in nodos))
    for n, fila in zip(nodos, A):
        print(f"{n:>5} " + "".join(f"{v:8.4f}" for v in fila))


def analizar(carpeta, ruta):
    print(f"\n==================== {carpeta} ====================")
    red = construir_diccionario(ruta)
    d = 0.85
    nodos, P, M = construir_matriz_google(red, d)
    m = len(nodos)
    assert np.allclose(P.sum(axis=1), 1), "P no es estocastica"
    assert np.allclose(M.sum(axis=1), 1), "M no es estocastica"

    imprimir_matriz("Matriz de transicion P (Dead Ends reparados):", P, nodos)
    print(f"\nMatriz de Google: M = d*P + (1-d)/m * E   con d = {d}, m = {m}")
    print(f"  (1-d)/m = {(1 - d) / m:.6f}")
    imprimir_matriz("Matriz de Google M:", M, nodos)
    print("\nSuma de cada fila de M:", np.round(M.sum(axis=1), 6))

    pi, hist = metodo_potencia(M)
    print("\nMetodo de la potencia: ")

    print(" Simulación Monte Carlo del Navegante Aleatorio")
    pi_mc = monte_carlo_surfer(M)

    print(f"\n{'Nodo':>4} | {'Potencia':>9} | {'M. Carlo':>9}")
    for n, a_, b_ in zip(nodos, pi, pi_mc):
        print(f"{n:>4} | {a_:9.5f} | {b_:9.5f}")

    grafica_convergencia(nodos, hist, carpeta)
    grafica_comparativa(nodos, pi, pi_mc, carpeta)
    grafica_sensibilidad(red, carpeta)
    grafica_ranking(nodos, pi, carpeta)
    print("\nGraficas guardadas como PNG. Cierra las ventanas para volver al menu.")
    plt.show()


def main():
    if not os.path.isdir(RUTA_HTML):
        print(f"No existe la carpeta {RUTA_HTML}")
        return

    carpetas = sorted(c for c in os.listdir(RUTA_HTML)
                      if os.path.isdir(os.path.join(RUTA_HTML, c)))
    if not carpetas:
        print("No hay carpetas de topologias dentro de htmls/")
        return

    while True:
        print("\n===== PAGERANK =====")
        for k, c in enumerate(carpetas, 1):
            print(f" {k}. Topologia {k} ({c})")
        print(" 0. Salir")
        op = input("Elige: ").strip()

        if op == "0":
            return
        elif op.isdigit() and 1 <= int(op) <= len(carpetas):
            c = carpetas[int(op) - 1]
            analizar(c, os.path.join(RUTA_HTML, c))
        else:
            print("Opcion no valida.")


if __name__ == "__main__":
    main()
