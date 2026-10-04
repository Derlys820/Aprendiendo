import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. EXTRACCIÓN DE ENLACES DESDE HTML
# ============================================================

def extraer_enlaces(carpeta):
    """
    Extrae enlaces de todos los archivos HTML en una carpeta.
    Usa expresiones regulares para encontrar <a href="...">.
    """
    red = {}
    patron = re.compile(r'<a\s+href=["\']([^"\']+)["\']')
    
    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.endswith('.html'):
            continue
        
        nodo = archivo.replace('.html', '')
        ruta = os.path.join(carpeta, archivo)
        
        with open(ruta, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        enlaces = []
        for match in patron.findall(contenido):
            destino = match.replace('.html', '')
            enlaces.append(destino)
        
        red[nodo] = enlaces
    
    return red

# ============================================================
# 2. CONSTRUCCIÓN DE LA MATRIZ DE GOOGLE
# ============================================================

def construir_matriz_google(diccionario_red, d=0.85):
    """
    Construye P y M a partir del diccionario de red.
    
    Parámetros:
        diccionario_red (dict): {nodo: [destinos]}
        d (float): factor de amortiguación
    
    Retorna:
        nodos (list): lista ordenada de nodos
        P (np.array): matriz de transición
        M (np.array): Matriz de Google
    """
    nodos = sorted(list(diccionario_red.keys()))
    m = len(nodos)
    idx = {nodo: i for i, nodo in enumerate(nodos)}
    P = np.zeros((m, m))
    
    for nodo, enlaces in diccionario_red.items():
        i = idx[nodo]
        
        if len(enlaces) == 0:
            # Reparación de Dead End
            P[i, :] = 1.0 / m
        else:
            total_enlaces = len(enlaces)  # Incluye duplicados
            for destino in enlaces:
                j = idx[destino]
                P[i, j] += 1.0 / total_enlaces
    
    E = np.ones((m, m))
    M = d * P + (1.0 - d) * (1.0 / m) * E
    
    return nodos, P, M

# ============================================================
# 3. MÉTODO DE LA POTENCIA
# ============================================================

def metodo_potencia(M, tol=1e-8, max_iter=500):
    """
    Resuelve pi * M = pi por el método de la potencia.
    """
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

# ============================================================
# 4. SIMULACIÓN MONTE CARLO
# ============================================================

def monte_carlo_surfer(M, pasos=150000):
    """
    Simula el navegante aleatorio.
    """
    m = M.shape[0]
    estado = np.random.randint(0, m)
    visitas = np.zeros(m)
    
    for _ in range(pasos):
        visitas[estado] += 1
        estado = np.random.choice(m, p=M[estado, :])
    
    return visitas / pasos

# ============================================================
# 5. GRÁFICAS
# ============================================================

def graficar_convergencia(historial, nodos, titulo, carpeta_salida):
    """Grafica la convergencia del método de la potencia."""
    plt.figure(figsize=(10, 6))
    for i, nodo in enumerate(nodos):
        plt.plot(historial[:, i], label=nodo, linewidth=2)
    plt.xlabel('Iteración')
    plt.ylabel('Probabilidad')
    plt.title(f'Convergencia - {titulo}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, f'convergencia_{titulo}.png'), dpi=150)
    plt.close()

def graficar_comparativa(pi_num, pi_mc, nodos, titulo, carpeta_salida):
    """Compara método numérico vs Monte Carlo."""
    x = np.arange(len(nodos))
    ancho = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - ancho/2, pi_num, ancho, label='Método Numérico', alpha=0.8)
    plt.bar(x + ancho/2, pi_mc, ancho, label='Monte Carlo', alpha=0.8)
    plt.xlabel('Página')
    plt.ylabel('PageRank')
    plt.title(f'Comparativa - {titulo}')
    plt.xticks(x, nodos)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, f'comparativa_{titulo}.png'), dpi=150)
    plt.close()

def graficar_ranking(pi, nodos, titulo, carpeta_salida):
    """Grafica el ranking final de PageRank."""
    plt.figure(figsize=(12, 6))
    plt.bar(nodos, pi, color='steelblue', alpha=0.8)
    plt.xlabel('Página')
    plt.ylabel('PageRank')
    plt.title(f'Ranking - {titulo}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, f'ranking_{titulo}.png'), dpi=150)
    plt.close()

def graficar_sensibilidad(red, d_valores, carpeta_salida):
    """Analiza cómo cambia la convergencia según d."""
    iteraciones = []
    for d in d_valores:
        nodos, P, M = construir_matriz_google(red, d)
        _, historial = metodo_potencia(M)
        iteraciones.append(len(historial))
    
    plt.figure(figsize=(10, 6))
    plt.plot(d_valores, iteraciones, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Factor de amortiguación d')
    plt.ylabel('Iteraciones hasta convergencia')
    plt.title('Sensibilidad al factor d')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, 'sensibilidad_d.png'), dpi=150)
    plt.close()

# ============================================================
# 6. PIPELINE POR TOPOLOGÍA
# ============================================================

def analizar_topologia(carpeta_html, nombre, carpeta_salida, d=0.85):
    """
    Ejecuta todo el pipeline para una topología.
    """
    print(f"\n{'='*60}")
    print(f"ANÁLISIS: {nombre}")
    print(f"{'='*60}")
    
    # 1. Extraer enlaces de HTML
    red = extraer_enlaces(carpeta_html)
    print(f"\nRed extraída: {red}")
    
    # 2. Construir P y M
    nodos, P, M = construir_matriz_google(red, d)
    print(f"\nNodos: {nodos}")
    print(f"\nMatriz P:\n{P}")
    print(f"\nMatriz M (d={d}):\n{M}")
    
    # 3. Método de potencia
    pi_num, historial = metodo_potencia(M)
    print(f"\nPageRank (numérico):")
    for nodo, valor in zip(nodos, pi_num):
        print(f"  {nodo}: {valor:.4f}")
    print(f"Iteraciones: {len(historial) - 1}")
    
    # 4. Monte Carlo
    pi_mc = monte_carlo_surfer(M, pasos=150000)
    print(f"\nPageRank (Monte Carlo):")
    for nodo, valor in zip(nodos, pi_mc):
        print(f"  {nodo}: {valor:.4f}")
    
    # 5. Comparación
    diff = np.linalg.norm(pi_num - pi_mc, 1)
    print(f"\nDiferencia L1: {diff:.6f}")
    
    # 6. Gráficas
    graficar_convergencia(historial, nodos, nombre, carpeta_salida)
    graficar_comparativa(pi_num, pi_mc, nodos, nombre, carpeta_salida)
    graficar_ranking(pi_num, nodos, nombre, carpeta_salida)
    
    return pi_num, pi_mc, nodos

# ============================================================
# 7. EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Crear carpeta de resultados
    carpeta_salida = 'resultados'
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Analizar las tres topologías
    pi1, mc1, nodos1 = analizar_topologia(
        'paginas/topologia1', 'Topologia_1', carpeta_salida
    )
    pi2, mc2, nodos2 = analizar_topologia(
        'paginas/topologia2', 'Topologia_2', carpeta_salida
    )
    pi3, mc3, nodos3 = analizar_topologia(
        'paginas/topologia3', 'Topologia_3', carpeta_salida
    )
    
    # Análisis de sensibilidad con Topología 1
    red1 = extraer_enlaces('paginas/topologia1')
    d_valores = [0.10, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99]
    graficar_sensibilidad(red1, d_valores, carpeta_salida)
    
    print("\n✅ Laboratorio completado. Gráficas en carpeta 'resultados/'.")