import numpy as np

print("=" * 65)
print("EJECUTANDO LABORATORIO: REDES NEURONALES DENSAS (SESIÓN 12)")
print("=" * 65)

# Función de Activación Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# -------------------------------------------------------------
# PARTE 1: Procesamiento de 1 solo cliente
# -------------------------------------------------------------
print("\n--- PARTE 1: PROCESAMIENTO DE 1 CLIENTE ---")
X_single = np.array([0.5, 0.8, 0.2])

# Pesos y Sesgos Capa Oculta (3x4)
W1 = np.array([
    [0.1,  0.2, 0.3,  0.4],
    [-0.5, 0.6, 0.7, -0.8],
    [0.9, -0.1, 0.2,  0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])

# Proceso Capa Oculta
Z1_single = np.dot(X_single, W1) + b1
A1_single = sigmoide(Z1_single)

# Pesos y Sesgos Capa Salida (4x1)
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# Proceso Capa Salida
Z2_single = np.dot(A1_single, W2) + b2
Salida_single = sigmoide(Z2_single)

print(f"Valores Z1 (Combinación lineal sin activar) : {np.round(Z1_single, 4)}")
print(f"Valores A1 (Activación Sigmoide [0, 1])      : {np.round(A1_single, 4)}")
print(f"Probabilidad Final Crédito Cliente 1         : {np.round(Salida_single[0], 4)}")

# -------------------------------------------------------------
# PARTE 2: Reto Dimensional - Lote de 2 Clientes (Batch)
# -------------------------------------------------------------
print("\n--- PARTE 2: RETO DIMENSIONAL (BATCH DE 2 CLIENTES) ---")
X_batch = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# Propagación Matricial en Lote
Z1_batch = np.dot(X_batch, W1) + b1
A1_batch = sigmoide(Z1_batch)

Z2_batch = np.dot(A1_batch, W2) + b2
Salida_batch = sigmoide(Z2_batch)

print("Matriz A1 (Salidas de Capa Oculta para 2 Clientes):")
print(np.round(A1_batch, 4))

print("\nPredicción de la Red (Probabilidades por Cliente):")
for i, prob in enumerate(Salida_batch, 1):
    print(f" -> Cliente {i}: {np.round(prob, 4)} ({'Aprobado' if prob >= 0.5 else 'Rechazado'})")

print("\n✅ El cálculo vectorial/tensorial procesó ambos clientes simultáneamente sin alterar los pesos W.")
print("=" * 65)
