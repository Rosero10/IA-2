import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

print("=" * 65)
print("EJECUTANDO LABORATORIO: CLASIFICADOR UNIVERSAL KNN (SESIÓN 9)")
print("=" * 65)

# 1. Dataset ampliado con 10 clientes y 3 características: [Edad, Salario, Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],  # 0: NO COMPRA
    [22, 25, 0],  # 0: NO COMPRA
    [25, 32, 1],  # 0: NO COMPRA
    [28, 38, 2],  # 0: NO COMPRA
    [40, 50, 2],  # 1: COMPRA
    [35, 45, 1],  # 1: COMPRA
    [45, 60, 3],  # 1: COMPRA
    [50, 65, 2],  # 1: COMPRA
    [38, 48, 1],  # 1: COMPRA
    [23, 28, 0]   # 0: NO COMPRA
])

# Etiquetas: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])

# Nuevo cliente a evaluar: [30 años, $40k salario, 1 hijo]
nuevo_cliente = np.array([[30, 40, 1]])

# 2. Experimentos con K=1 y K=5
knn_k1 = KNeighborsClassifier(n_neighbors=1)
knn_k1.fit(X_entrenamiento, Y_entrenamiento)
pred_k1 = knn_k1.predict(nuevo_cliente)[0]

knn_k5 = KNeighborsClassifier(n_neighbors=5)
knn_k5.fit(X_entrenamiento, Y_entrenamiento)
pred_k5 = knn_k5.predict(nuevo_cliente)[0]

etiquetas_texto = {0: "NO COMPRA", 1: "COMPRA"}

print(f"\nEvaluando nuevo cliente: Edad={nuevo_cliente[0][0]}, Salario={nuevo_cliente[0][1]}k, Hijos={nuevo_cliente[0][2]}")
print(f" -> Predicción con K=1 : {pred_k1} ({etiquetas_texto[pred_k1]})")
print(f" -> Predicción con K=5 : {pred_k5} ({etiquetas_texto[pred_k5]})")

# 3. Visualización en espacio 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar datos de entrenamiento
no_compra = X_entrenamiento[Y_entrenamiento == 0]
compra = X_entrenamiento[Y_entrenamiento == 1]

ax.scatter(no_compra[:, 0], no_compra[:, 1], no_compra[:, 2], color='red', s=80, label='NO COMPRA (0)')
ax.scatter(compra[:, 0], compra[:, 1], compra[:, 2], color='green', s=80, label='COMPRA (1)')
ax.scatter(nuevo_cliente[0, 0], nuevo_cliente[0, 1], nuevo_cliente[0, 2], color='blue', s=150, marker='*', label='Nuevo Cliente')

ax.set_xlabel('Edad')
ax.set_ylabel('Salario (miles)')
ax.set_zlabel('N° Hijos')
ax.set_title('Clasificador KNN 3D: Proximidad de Clientes')
ax.legend()

plt.tight_layout()
plt.savefig('knn_clasificador_3d.png')

print("\n✅ Proceso de clasificación completado.")
print("📷 Visualización 3D guardada en: 'sesion_9/knn_clasificador_3d.png'")

# 4. Análisis de "La Maldición de la Dimensionalidad"
print("\n💡 ANÁLISIS DE LA MALDICIÓN DE LA DIMENSIONALIDAD:")
print("   - Si aumentamos de 3 a 1,000 dimensiones (ej. píxeles de una imagen):")
print("   - El volumen del espacio crece exponencialmente, haciendo que todos los puntos")
print("     queden 'lejos' unos de otros y a distancias muy similares entre sí.")
print("   - La Distancia Euclidiana pierde su capacidad discriminativa porque la diferencia")
print("     relativa entre el vecino más cercano y el más lejano tiende a cero.")
print("   - Solución: Aplicar reducción de dimensionalidad (ej. PCA) o extracción de características")
print("     relevantes antes de ejecutar KNN.")
print("=" * 65)
