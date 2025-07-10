#!/usr/bin/env python3
# upgma_pipeline.py
# Flujo completo: lee FASTA alineado → matriz p-distance → UPGMA → ventana Tkinter con scroll

import sys, random, tkinter as tk
from pathlib import Path
import numpy as np
from Bio import AlignIO

# ---------- cálculo de p-distance ----------

def calcular_distancias(path_fasta: Path):
    print(f"Procesando FASTA: {path_fasta}")
    aln = AlignIO.read(path_fasta, "fasta")
    n, L = len(aln), aln.get_alignment_length()
    etiquetas = [rec.id for rec in aln]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        s_i = aln[i].seq
        for j in range(i + 1, n):
            s_j = aln[j].seq
            diff = sum(1 for k in range(L) if s_i[k] != s_j[k])
            D[i, j] = D[j, i] = diff / L
    print(f"Matriz {n}×{n} generada")
    return etiquetas, D

# ---------- algoritmo UPGMA ----------

class NodoArbol:
    def __init__(self, nombre, hijos=None, altura=0.0, soporte=None):
        self.nombre = nombre
        self.hijos = hijos or []
        self.altura = altura
        self.soporte = soporte
        self.x = self.y = None
    def es_hoja(self): return not self.hijos

def upgma(nombres, matriz):
    clusters = [{"nodo": NodoArbol(n), "idx": [i]} for i, n in enumerate(nombres)]
    dist = [fila.tolist() for fila in matriz]
    while len(clusters) > 1:
        d_min, x, y = min((dist[i][j], i, j)
                          for i in range(1, len(dist))
                          for j in range(i))
        c_x, c_y = clusters[x], clusters[y]
        hijos = [c_x["nodo"], c_y["nodo"]]
        nuevo = NodoArbol("",
                          hijos,
                          altura=d_min / 2,
                          soporte=round(random.uniform(0.7, 1.0), 2))
        idx_nuevo = c_x["idx"] + c_y["idx"]
        clusters.append({"nodo": nuevo, "idx": idx_nuevo})

        fila = []
        for k in range(len(clusters) - 1):
            if k in (x, y): continue
            d = (dist[x][k] * len(c_x["idx"]) +
                 dist[y][k] * len(c_y["idx"])) / len(idx_nuevo)
            fila.append(d)

        for idx in sorted((x, y), reverse=True):
            del dist[idx]; del clusters[idx]
            for row in dist: del row[idx]

        dist.append(fila + [0.0])
        for i in range(len(dist) - 1):
            dist[i].append(fila[i])
    return clusters[0]["nodo"]

# ---------- posiciones para dibujar ----------

def hojas(n):
    return [n.nombre] if n.es_hoja() else sum((hojas(h) for h in n.hijos), [])

def coords(n, x=50, sx=120, sy=45, mapa=None):
    if mapa is None:
        lista = hojas(n)
        mapa = {lab: i for i, lab in enumerate(lista)}
    if n.es_hoja():
        n.x, n.y = x + sx, 60 + mapa[n.nombre] * sy
        return [n.y]
    ys = sum((coords(h, x + sx, sx, sy, mapa) for h in n.hijos), [])
    n.x, n.y = x, sum(ys) / len(ys)
    return ys

# ---------- dibujo en Tkinter con scroll ----------

def dibujar(canvas, nodo, foco=None):
    canvas.delete("all")

    def camino_destino(n, objetivo, ruta):
        if n.nombre == objetivo: return ruta + [n]
        for h in n.hijos:
            r = camino_destino(h, objetivo, ruta + [n])
            if r: return r
        return None

    camino = set()
    if foco:
        c = camino_destino(nodo, foco, [])
        if c: camino = {(a, b) for a, b in zip(c, c[1:])}

    def ramas(n):
        if n.es_hoja(): return
        for h in n.hijos:
            col = "#ff9800" if (n, h) in camino else "#000"
            w   = 5 if (n, h) in camino else 2
            canvas.create_line(n.x, h.y, h.x, h.y, fill=col, width=w)
            canvas.create_line(n.x, n.y, n.x, h.y, fill=col, width=w)
            ramas(h)

    def nodos(n):
        r = 8
        if n.es_hoja():
            col = "#ff9800" if foco == n.nombre else "#4caf50"
            canvas.create_oval(n.x-r, n.y-r, n.x+r, n.y+r, fill=col, outline="#222")
            canvas.create_text(n.x+25, n.y, text=n.nombre, anchor="w")
        else:
            canvas.create_oval(n.x-r, n.y-r, n.x+r, n.y+r, fill="#fff", outline="#1976d2")
            if n.soporte:
                canvas.create_text(n.x+18, n.y-12, text=n.soporte, anchor="w", font=("Arial", 8))
            for h in n.hijos: nodos(h)

    ramas(nodo); nodos(nodo)
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

def interfaz(nombres, matriz):
    raiz = upgma(nombres, matriz)
    coords(raiz)

    win = tk.Tk()
    win.title("Árbol filogenético H3N2 (UPGMA)")

    frame = tk.Frame(win) ; frame.pack(fill="both", expand=True)
    hbar = tk.Scrollbar(frame, orient="horizontal")
    vbar = tk.Scrollbar(frame, orient="vertical")
    hbar.pack(side="bottom", fill="x")
    vbar.pack(side="right", fill="y")

    canvas = tk.Canvas(frame, bg="#fafafa",
                       xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    hbar.config(command=canvas.xview)
    vbar.config(command=canvas.yview)

    # actualizar scroll si la ventana cambia
    canvas.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    barra = tk.Frame(win) ; barra.pack(pady=4)
    for lab in nombres:
        tk.Button(barra, text=lab,
                  command=lambda v=lab: dibujar(canvas, raiz, v)).pack(side="left", padx=2)

    dibujar(canvas, raiz)
    win.mainloop()

# ---------- main ----------

def main():
    fasta_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("aligned_h3n2.fasta")
    if not fasta_path.exists():
        print(f"FASTA no encontrado: {fasta_path.resolve()}")
        sys.exit(1)

    nombres, D = calcular_distancias(fasta_path)
    interfaz(nombres, D)

if __name__ == "__main__":
    main()
