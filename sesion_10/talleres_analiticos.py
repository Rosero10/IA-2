import numpy as np
from sklearn.svm import SVC

print("=" * 65)
print("SOLUCIÓN TALLER ANALÍTICO - SESIÓN 10: MARGEN Y VECTORES DE SOPORTE")
print("=" * 65)

# Dataset base
X = np.array([
    [2, 2], [3, 3], [4, 2],  # Clase A (0)
    [6, 6], [7, 8], [8, 7]   # Clase B (1)
])
Y = np.array([0, 0, 0, 1, 1, 1])

# Entrenar SVM Lineal
modelo_svm = SVC(kernel='linear')
modelo_svm.fit(X, Y)

# 1. Vectores de soporte
vectores = modelo_svm.support_vectors_
print("\n1. Vectores de Soporte identificados por el modelo:")
for v in vectores:
    clase = "Clase A (0)" if v in X[:3] else "Clase B (1)"
    print(f"   - Coordenada: {v} -> {clase}")

# 2. Ecuación de la frontera óptima
w = modelo_svm.coef_[0]
b = modelo_svm.intercept_[0]
print(f"\n2. Ecuación matemática del hiperplano de separación:")
print(f"   {w[0]:.2f}*x + {w[1]:.2f}*y + ({b:.2f}) = 0  =>  x + y = 9 (y = -x + 9)")

# 3. Experimento: Agregar punto (1,1) a la Clase A
X_nuevo = np.vstack([X, [1, 1]])
Y_nuevo = np.append(Y, 0)

modelo_svm_nuevo = SVC(kernel='linear')
modelo_svm_nuevo.fit(X_nuevo, Y_nuevo)

vectores_nuevos = modelo_svm_nuevo.support_vectors_

print("\n3. Análisis al agregar el punto (1,1) a la Clase A:")
print(f"   - Nuevos vectores de soporte: \n{vectores_nuevos}")
print("\n💡 JUSTIFICACIÓN TEÓRICA:")
print("   - La posición de la línea NO cambia.")
print("   - Los modelos SVM construyen el hiperplano óptimo basándose EXCLUSIVAMENTE")
print("     en los vectores de soporte (los puntos más cercanos a la frontera).")
print("   - El punto (1,1) queda muy alejado del margen (en el interior de la Clase A),")
print("     por lo que su multiplicador de Lagrange es alpha_i = 0 y no altera la frontera.")
print("=" * 65)
