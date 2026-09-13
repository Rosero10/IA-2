cat << 'EOF' > talleres_analiticos.py
import numpy as np

print("=" * 65)
print("SOLUCIÓN TALLER ANALÍTICO - SESIÓN 4: CONVOLUCIÓN 2D (FILTRO MEDIA)")
print("=" * 65)

# Matriz 3x3 original con ruido de sal en el centro
I = np.array([
    [10,  20, 30],
    [15, 250, 15],
    [20,  10, 20]
], dtype=float)

# Kernel de media 3x3
K = np.ones((3, 3)) / 9.0

print("\nMatriz Original I (con ruido 250 en el centro):")
print(I)

# 1. Cálculo de Convolución en el píxel central
suma_productos = np.sum(I * K)
nuevo_valor_float = suma_productos
nuevo_valor_int = int(np.round(suma_productos))

print(f"\n1. Cálculo matemático del nuevo píxel central:")
print(f"   Sumatoria de productos (390 / 9) = {nuevo_valor_float:.2f}")
print(f"   Valor redondeado (entero de imagen) = {nuevo_valor_int}")

# 2. Justificación teórica
print("\n2. Explicación de por qué difumina o suaviza la imagen:")
print("   - El valor anómalo (250) fue 'repartido' y promediado con la vecindad oscura (~15).")
print("   - El píxel central bajó drásticamente de 250 a ~43, atenuando el brillo del pico de ruido.")
print("   - Sin embargo, los píxeles vecinos oscuros ahora se ven influenciados por el 250, lo que")
print("     genera un efecto visual de difuminado o 'borrosidad' gris en lugar de eliminarlo completamente.")
print("=" * 65)
EOF