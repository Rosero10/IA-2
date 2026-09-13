import numpy as np

print("=" * 60)
print("SOLUCIÓN TALLER ANALÍTICO 1 - SESIÓN 3: FUNCIÓN ESCALÓN")
print("=" * 60)

# Matriz 3x3 original dada en la guía
I = np.array([
    [80,  120, 140],
    [90,  200, 210],
    [50,  130, 250]
])

print("\nMatriz Original I:")
print(I)

# 1. Aplicación de Umbralización Binaria con T = 135
# f(x,y) = 255 si I(x,y) >= T else 0
T = 135
matriz_binaria = np.where(I >= T, 255, 0)

print(f"\n1. Matriz Resultante con T = {T}:")
print(matriz_binaria)

# 2. Explicación del Error y Efecto Visual
print("\n2. Análisis del Error al elegir T = 135:")
print("   - Si el objetivo era aislar valores > 100, los píxeles con valores 120 y 130")
print("     quedaron incorrectamente clasificados como fondo (0) en lugar de objeto (255).")
print("   - Efecto Visual: El objeto segmentado sufrirá pérdida de información/sub-segmentación,")
print("     apareciendo con 'huecos' o bordes recortados en las zonas con intensidades entre 101 y 134.")
print("-" * 60)