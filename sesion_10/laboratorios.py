import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

print("=" * 65)
print("EJECUTANDO LABORATORIO: FRONTERAS NO LINEALES (SESIÓN 10)")
print("=" * 65)

# 1. Dataset ampliado con el punto de prueba [5, 5] etiquetado como Clase A (0)
X_lab = np.array([
    [2, 2], [3, 3], [4, 2],  # Clase A
    [6, 6], [7, 8], [8, 7],  # Clase B
    [5, 5]                  # Punto "engañoso" perteneciente a Clase A
])
Y_lab = np.array([0, 0, 0, 1, 1, 1, 0])

# 2. Entrenamiento con Kernel Lineal vs Kernel RBF
modelo_lineal = SVC(kernel='linear')
modelo_lineal.fit(X_lab, Y_lab)

modelo_rbf = SVC(kernel='rbf', C=10.0, gamma='scale')
modelo_rbf.fit(X_lab, Y_lab)

# Evaluar punto de prueba [5, 4]
punto_prueba = np.array([[5, 4]])
pred_lin = modelo_lineal.predict(punto_prueba)[0]
pred_rbf = modelo_rbf.predict(punto_prueba)[0]

print(f"Predicción para el punto [5, 4]:")
print(f" -> Con Kernel Lineal: Clase {pred_lin}")
print(f" -> Con Kernel RBF   : Clase {pred_rbf}")

# 3. Malla para graficar las fronteras de decisión
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

plt.figure(figsize=(14, 6))

# Subplot 1: Kernel Lineal
plt.subplot(1, 2, 1)
Z_lin = modelo_lineal.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lin = Z_lin.reshape(xx.shape)
plt.contourf(xx, yy, Z_lin, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_lab[Y_lab==0][:, 0], X_lab[Y_lab==0][:, 1], color='blue', s=100, label='Clase A (0)', edgecolors='k')
plt.scatter(X_lab[Y_lab==1][:, 0], X_lab[Y_lab==1][:, 1], color='red', s=100, marker='x', label='Clase B (1)')
plt.scatter(5, 5, color='cyan', s=150, marker='o', edgecolors='black', label='Punto [5,5] Clase A')
plt.title("SVM con Kernel Lineal (Forzado)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Subplot 2: Kernel RBF
plt.subplot(1, 2, 2)
Z_rbf = modelo_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)
plt.contourf(xx, yy, Z_rbf, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_lab[Y_lab==0][:, 0], X_lab[Y_lab==0][:, 1], color='blue', s=100, label='Clase A (0)', edgecolors='k')
plt.scatter(X_lab[Y_lab==1][:, 0], X_lab[Y_lab==1][:, 1], color='red', s=100, marker='x', label='Clase B (1)')
plt.scatter(5, 5, color='cyan', s=150, marker='o', edgecolors='black', label='Punto [5,5] Clase A')
plt.title("SVM con Kernel RBF (Aislamiento No Lineal)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.tight_layout()
plt.savefig('svm_fronteras_resultado.png')

print("\n✅ Proceso completado exitosamente.")
print("📷 Comparativa gráfica guardada en: 'sesion_10/svm_fronteras_resultado.png'")

# 4. Reflexión teórica
print("\n💡 REFLEXIÓN - ESCENARIOS DEL MUNDO REAL DONDE FALLA EL KERNEL LINEAL:")
print("   1. IMÁGENES MÉDICAS / ONCOLOGÍA: Detección de tumores donde el tejido maligno")
print("      está concéntricamente rodeado de tejido sano (patrón en forma de dona).")
print("   2. RECONOCIMIENTO FACIAL: Cambios de iluminación o expresión generan agrupaciones")
print("      no separables mediante hiperplanos rectos.")
print("   3. DETECCIÓN DE ANOMALÍAS / FRAUDE: Transacciones fraudulentas mezcladas en clusters")
print("      específicos inmersos dentro del comportamiento normal del cliente.")
print("=" * 65)
