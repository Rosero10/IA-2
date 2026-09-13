import numpy as np

print("=" * 65)
print("SOLUCIÓN TALLER ANALÍTICO - SESIÓN 5: OPERADORES SOBEL (GRADIENTE)")
print("=" * 65)

# Matriz 3x3 con borde vertical perfecto (izquierda negra, derecha blanca)
I = np.array([
    [0, 0, 255],
    [0, 0, 255],
    [0, 0, 255]
], dtype=float)

# Kernels de Sobel
Gx_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

Gy_kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=float)

# 1. Convolución en el píxel central con Sobel X
gx_val = np.sum(I * Gx_kernel)

# 2. Convolución en el píxel central con Sobel Y
gy_val = np.sum(I * Gy_kernel)

# Magnitud del gradiente total
magnitud = np.sqrt(gx_val**2 + gy_val**2)

print("\n1. Resultado Sobel X (Gx - Bordes Verticales):")
print(f"   Gx = {gx_val:.2f}")

print("\n2. Resultado Sobel Y (Gy - Bordes Horizontales):")
print(f"   Gy = {gy_val:.2f}")

print(f"\n3. Magnitud del Gradiente Total (G = sqrt(Gx^2 + Gy^2)):")
print(f"   G = {magnitud:.2f}")

print("\n💡 ANÁLISIS TEÓRICO:")
print("   - Gx dio 1020 (un valor muy alto), lo que confirma un cambio brusco de intensidad")
print("     en sentido horizontal (borde vertical).")
print("   - Gy dio 0 porque la intensidad no cambia al desplazarnos verticalmente (de arriba a abajo).")
print("   - Esto indica que el borde es estrictamente vertical (perpendicular al eje X).")
print("=" * 65)
