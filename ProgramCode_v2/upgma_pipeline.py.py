#!/usr/bin/env python3
"""
upgma_pipeline.py
Autor     : Jason Ayosa, Rodrigo Figueroa
Fecha     : 2025-07-17
Propósito : Construir y visualizar un árbol filogenético de H3N2 usando UPGMA y p-distance.

Requisitos
----------
Python ≥ 3.8
pip install biopython numpy
Tkinter (incluido en la instalación estándar de Python)
"""

import sys
import random
import tkinter as tk
from pathlib import Path

import numpy as np
from Bio import AlignIO


# ───────────────────────────────────────────────────────────────
# FUNCIÓN: calcular_distancias
# ───────────────────────────────────────────────────────────────
def calcular_distancias(path_fasta: Path):
    """
    Calcula la matriz p-distance a partir de un archivo FASTA alineado

    Parámetros
    ----------
    path_fasta : Path
        Ruta del archivo FASTA con todas las secuencias ya alineadas

    Retorna
    -------
    tuple[list[str], np.ndarray]
        • Lista de identificadores en el mismo orden que aparecen en el FASTA
        • Matriz cuadrada (n x n) con las distancias p-distance. 

    Complejidad
    -----------
    Tiempo : O(n² · L),  donde n = número de secuencias, L = longitud de la alineación
    Memoria: O(n²)      para almacenar la matriz completa.
    """
    print(f"Procesando FASTA: {path_fasta}")
    alignment = AlignIO.read(path_fasta, "fasta")
    n = len(alignment)
    L = alignment.get_alignment_length()

    # Extraer IDs para etiquetar las hojas del árbol.
    etiquetas = [rec.id for rec in alignment]

    # Crear matriz n×n inicializada en ceros (dtype más eficiente).
    D = np.zeros((n, n), dtype=np.float32)

    # Comparar cada par de secuencias (i < j) y calcular p-distance.
    for i in range(n):
        seq_i = alignment[i].seq
        for j in range(i + 1, n):
            seq_j = alignment[j].seq
            # Contar posiciones diferentes entre las dos secuencias.
            diff = sum(1 for k in range(L) if seq_i[k] != seq_j[k])
            distancia = diff / L
            # Asignar valor simétrico en la matriz.
            D[i, j] = D[j, i] = distancia

    print(f"Matriz {n}x{n} generada")
    return etiquetas, D


# ───────────────────────────────────────────────────────────────
# CLASE: NodoArbol
# ───────────────────────────────────────────────────────────────
class NodoArbol:
    """
    Nodo para representar un clúster en UPGMA

    Atributos
    ---------
    nombre  : str
        Etiqueta de la hoja (vacío en nodos internos)
    hijos   : list[NodoArbol]
        Lista de nodos descendientes.
    altura  : float
        Distancia al ancestro común dividida entre 2
    soporte : float
        Valor simulando bootstrap (0.70–1.00)
    x, y    : int
        Coordenadas para dibujo en el canvas
    """

    
    def __init__(self, nombre, hijos=None, altura=0.0, soporte=None):
        """
        Inicializa un nodo del árbol filogenético.

        Parámetros
        ----------
        nombre : str
            Etiqueta asignada a este nodo (vacía para nodos internos).
        hijos : list[NodoArbol], opcional
            Lista de nodos hijos; si no se provee, se considera hoja.
        altura : float, opcional
            Distancia entre este nodo y sus hijos dividida entre dos.
        soporte : float, opcional
            Valor de soporte bootstrap simulado (rango 0.7–1.0).
        """
        self.nombre = nombre
        self.hijos = hijos or []
        self.altura = altura
        self.soporte = soporte
        self.x = self.y = None 

    def es_hoja(self):
        """
        Indica si el nodo es una hoja (sin descendientes).

        Retorna
        -------
        bool
        True si no tiene hijos, False en caso contrario.
        """
        # Devuelve True si el nodo no tiene hijos
        return not self.hijos


# ───────────────────────────────────────────────────────────────
# FUNCIÓN: upgma
# ───────────────────────────────────────────────────────────────
def upgma(nombres, matriz):
    """
    Construye el árbol UPGMA a partir de la matriz de distancias

    Parámetros
    ----------
    nombres : list[str]
        Lista de IDs de secuencias
    matriz : np.ndarray
        Matriz simétrica de distancias

    Retorna
    -------
    NodoArbol
        Raíz del árbol filogenético construido
    """
    # Inicializar un clúster por secuencia.
    clusters = [{"nodo": NodoArbol(n), "idx": [i]} for i, n in enumerate(nombres)]
    dist = [fila.tolist() for fila in matriz]

    # Fusionar iterativamente hasta un único clúster.
    while len(clusters) > 1:
        # Encontrar par con distancia mínima.
        d_min, x, y = min(
            (dist[i][j], i, j)
            for i in range(1, len(dist))
            for j in range(i)
        )
        cx, cy = clusters[x], clusters[y]
        hijos = [cx["nodo"], cy["nodo"]]

        # Crear nodo interno con altura = d_min / 2.
        nuevo = NodoArbol(
            nombre="",
            hijos=hijos,
            altura=d_min / 2,
            soporte=round(random.uniform(0.70, 1.00), 2)
        )
        idx_nuevo = cx["idx"] + cy["idx"]
        clusters.append({"nodo": nuevo, "idx": idx_nuevo})

        # Calcular distancias promedio al nuevo clúster.
        fila = []
        for k in range(len(clusters) - 1):
            if k in (x, y):
                continue
            d = (
                dist[x][k] * len(cx["idx"])
                + dist[y][k] * len(cy["idx"])
            ) / len(idx_nuevo)
            fila.append(d)

        # Elimina las filas/columnas de los clústeres ya fusionados
        for idx in sorted((x, y), reverse=True):
            del dist[idx]
            del clusters[idx]
            for row in dist:
                del row[idx]

        # Añade la nueva fila/columna al final de la matriz
        dist.append(fila + [0.0])
        for i in range(len(dist) - 1):
            dist[i].append(fila[i])

    # Retorna la raíz final
    return clusters[0]["nodo"]


# ───────────────────────────────────────────────────────────────
# FUNCIÓN: hojas (para obtener las etiquetas de hojas)
# ───────────────────────────────────────────────────────────────
def hojas(nodo):
    """
    Obtiene la lista de etiquetas de todas las hojas bajo un nodo dado.

    Parámetro
    ---------
    nodo : NodoArbol

    Retorna
    -------
    list[str]
    """
    return [nodo.nombre] if nodo.es_hoja() else sum((hojas(h) for h in nodo.hijos), [])


# ───────────────────────────────────────────────────────────────
# FUNCIÓN: coords
# ───────────────────────────────────────────────────────────────
def coords(nodo, x=50, sx=120, sy=45, mapa=None):
    """
    Asigna coordenadas (x, y) a cada nodo para su posición en el canvas.

    Parámetros
    ----------
    nodo : NodoArbol
    x    : int - desplazamiento horizontal inicial
    sx   : int - separación horizontal entre niveles
    sy   : int - separación vertical entre hojas
    mapa : dict - mapeo etiqueta -> fila (se genera la primera vez)
    """
    if mapa is None:
        lista = hojas(nodo)
        mapa = {lab: i for i, lab in enumerate(lista)}

    if nodo.es_hoja():
        nodo.x = x + sx
        nodo.y = 60 + mapa[nodo.nombre] * sy
        return [nodo.y]

    ys = sum((coords(h, x + sx, sx, sy, mapa) for h in nodo.hijos), [])
    nodo.x, nodo.y = x, sum(ys) / len(ys)
    return ys


# ───────────────────────────────────────────────────────────────
# FUNCIÓN: dibujar (pinta el árbol en el canvas)
# ───────────────────────────────────────────────────────────────
def dibujar(canvas, nodo, foco=None):
    """
    Dibuja el árbol en el canvas de Tkinter.

    Parámetros
    ----------
    canvas : tk.Canvas
        Lienzo donde se pintará el árbol.
    nodo   : NodoArbol
        Raíz del árbol.
    foco   : str | None
        Etiqueta de hoja que se desea resaltar (es opcional).
    """
    canvas.delete("all")

    # Determina el camino a la hoja 'foco' si se especifica.
    def camino_destino(n, objetivo, ruta):
        if n.nombre == objetivo:
            return ruta + [n]
        for h in n.hijos:
            r = camino_destino(h, objetivo, ruta + [n])
            if r:
                return r
        return None

    camino = set()
    if foco:
        c = camino_destino(nodo, foco, [])
        if c:
            camino = {(a, b) for a, b in zip(c, c[1:])}

    # Dibuja ramas horizontales y verticales.
    def ramas(n):
        if n.es_hoja():
            return
        for h in n.hijos:
            col = "#ff9800" if (n, h) in camino else "#000000"
            w   = 5 if (n, h) in camino else 2
            canvas.create_line(n.x, h.y, h.x, h.y, fill=col, width=w)
            canvas.create_line(n.x, n.y, n.x, h.y, fill=col, width=w)
            ramas(h)

    # Dibuja círculos en nodos y etiquetas de hojas.
    def nodos(n):
        r = 8
        if n.es_hoja():
            col = "#ff9800" if foco == n.nombre else "#4caf50"
            canvas.create_oval(n.x - r, n.y - r, n.x + r, n.y + r,
                               fill=col, outline="#222")
            canvas.create_text(n.x + 25, n.y, text=n.nombre, anchor="w")
        else:
            canvas.create_oval(n.x - r, n.y - r, n.x + r, n.y + r,
                               fill="#ffffff", outline="#1976d2")
            if n.soporte is not None:
                canvas.create_text(n.x + 18, n.y - 12,
                                   text=n.soporte, anchor="w", font=("Arial", 8))
            for h in n.hijos:
                nodos(h)

    ramas(nodo)
    nodos(nodo)

    # Ajusta la región de scrolling al tamaño del dibujo.
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))


# ───────────────────────────────────────────────────────────────
# FUNCIÓN: interfaz
# ───────────────────────────────────────────────────────────────
def interfaz(nombres, matriz):
    """
    Crea la ventana principal con canvas y barras de scroll vertical, incluyendo botones para resaltar rutas de hojas específicas, que serían las cepas del virus.
    """
    raiz = upgma(nombres, matriz)
    coords(raiz)

    win = tk.Tk()
    win.title("Árbol filogenético H3N2 (UPGMA)")

    # Contenedor con canvas y scrollbars
    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True)

    hbar = tk.Scrollbar(frame, orient="horizontal")
    vbar = tk.Scrollbar(frame, orient="vertical")
    hbar.pack(side="bottom", fill="x")
    vbar.pack(side="right", fill="y")

    canvas = tk.Canvas(frame, bg="#fafafa",
                       xscrollcommand=hbar.set,
                       yscrollcommand=vbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    hbar.config(command=canvas.xview)
    vbar.config(command=canvas.yview)

    # Actualizar scrollregion cuando cambie el tamaño
    canvas.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Barra de botones para resaltar cada cepa
    barra = tk.Frame(win)
    barra.pack(pady=4)
    for lab in nombres:
        tk.Button(barra,
                  text=lab,
                  command=lambda v=lab: dibujar(canvas, raiz, v)
                  ).pack(side="left", padx=2)

    dibujar(canvas, raiz)
    win.mainloop()


# ───────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL
# ───────────────────────────────────────────────────────────────
def main():
    """
    Punto de entrada: lee FASTA alineado (por defecto aligned_h3n2.fasta
    o ruta pasada como argumento) y lanza la interfaz.
    """
    fasta_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("aligned_h3n2.fasta")
    if not fasta_path.exists():
        print(f"ERROR: FASTA no encontrado: {fasta_path.resolve()}")
        sys.exit(1)

    nombres, D = calcular_distancias(fasta_path)
    interfaz(nombres, D)


if __name__ == "__main__":
    main()
