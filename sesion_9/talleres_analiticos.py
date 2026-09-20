import numpy as np

print("=" * 65)
print("SOLUCIÓN TALLER ANALÍTICO - SESIÓN 9: ALGORITMO KNN")
print("=" * 65)

# Coordenadas de los clientes historicos: [Edad, Salario]
A = np.array([20, 30])  # NO COMPRA (0)
B = np.array([40, 50])  # COMPRA (1)
C = np.array([35, 45])  # COMPRA (1)

N = np.array([30, 40])  # Nuevo cliente a clasificar

# 1. Calculo de Distancias Euclidianas
dist_A = np.linalg.norm(N - A)
dist_B = np.linalg.norm(N - B)
dist_C = np.linalg.norm(N - C)

print("\n1. Distancias Euclidianas desde el Nuevo Cliente N(30, 40):")
print(f"   - Distancia a A(20, 30) [NO COMPRA]: sqrt(200) = {dist_A:.2f}")
print(f"   - Distancia a B(40, 50) [COMPRA]   : sqrt(200) = {dist_B:.2f}")
print(f"   - Distancia a C(35, 45) [COMPRA]   : sqrt(50)  = {dist_C:.2f}")

# 2. Clasificacion con K=1
print("\n2. Clasificación con K = 1:")
print(f"   - El vecino más cercano es C (distancia = {dist_C:.2f}).")
print("   - Clase asignada: COMPRA (1)")

# 3. Clasificacion con K=3
print("\n3. Clasificación con K = 3:")
print("   - Los 3 vecinos consultados son C (7.07), A (14.14) y B (14.14).")
print("   - Votos registrados: C -> COMPRA (1), B -> COMPRA (1), A -> NO COMPRA (0).")
print("   - Resultado de votación: 2 votos COMPRA vs 1 voto NO COMPRA.")
print("   - Clase asignada: COMPRA (1)")
print("   - ¿Hubo cambio en la decisión?: NO. En ambos casos la predicción es COMPRA.")
print("=" * 65)
