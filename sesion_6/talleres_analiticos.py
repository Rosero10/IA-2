import numpy as np

print("=" * 65)
print("SOLUCIÓN TALLER ANALÍTICO - SESIÓN 6: BOUNDING BOX")
print("=" * 65)

# Coordenadas del contorno (X, Y)
puntos = np.array([
    [2, 4],   # Punto A
    [8, 2],   # Punto B
    [10, 7],  # Punto C
    [3, 9]    # Punto D
])

print("\nPuntos de la figura (X, Y):")
print("A(2,4), B(8,2), C(10,7), D(3,9)")

# 1. Determinar valores extremos (X_min, Y_min, X_max, Y_max)
x_min = np.min(puntos[:, 0])
y_min = np.min(puntos[:, 1])
x_max = np.max(puntos[:, 0])
y_max = np.max(puntos[:, 1])

# 2. Calcular Ancho (W) y Alto (H)
w = x_max - x_min
h = y_max - y_min

print("\n1. Coordenadas extremas del Bounding Box:")
print(f"   X_min = {x_min}")
print(f"   Y_min = {y_min}")
print(f"   X_max = {x_max}")
print(f"   Y_max = {y_max}")

print(f"\n2. Dimensiones de la caja delimitadora:")
print(f"   Ancho (W = X_max - X_min) = {w}")
print(f"   Alto  (H = Y_max - Y_min) = {h}")

print(f"\n   Esquina superior izquierda (X, Y): ({x_min}, {y_min})")
print(f"   Área de la caja envolvente: {w * h} unidades cuadradas")
print("=" * 65)
