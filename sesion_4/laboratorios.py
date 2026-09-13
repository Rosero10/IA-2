import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("EJECUTANDO LABORATORIO: ESTRATEGIAS DE SUAVIZADO (SESIÓN 4)")
print("=" * 65)

# 1. Crear una imagen de prueba limpia
imagen_base = np.zeros((300, 500), dtype=np.uint8)
cv2.putText(imagen_base, "IA-2", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, 255, 12)

# 2. Agregar Ruido de Sal y Pimienta agresivo
ruido_sal = np.random.choice([0, 255], size=(300, 500), p=[0.95, 0.05]).astype(np.uint8)
ruido_pimienta = np.random.choice([0, 255], size=(300, 500), p=[0.95, 0.05]).astype(np.uint8)

imagen_ruidosa = cv2.bitwise_or(imagen_base, ruido_sal)
imagen_ruidosa = cv2.bitwise_and(imagen_ruidosa, cv2.bitwise_not(ruido_pimienta))

# 3. Aplicar Filtros con Kernel 7x7
ksize = (7, 7)
blur_media = cv2.blur(imagen_ruidosa, ksize)
blur_gauss = cv2.GaussianBlur(imagen_ruidosa, ksize, 0)
blur_mediana = cv2.medianBlur(imagen_ruidosa, 7)

# 4. Generar reporte comparativo en matplotlib
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.title("1. Imagen Original con Ruido Sal y Pimienta")
plt.imshow(imagen_ruidosa, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("2. Filtro de Media (7x7) - Difuminado/Manchas")
plt.imshow(blur_media, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("3. Filtro Gaussiano (7x7) - Bordes Suaves")
plt.imshow(blur_gauss, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("4. Filtro de Mediana (7x7) - Eliminación Perfecta")
plt.imshow(blur_mediana, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('filtrado_convolucion_resultado.png')

print("\n✅ Proceso de filtrado completado.")
print("📷 Se generó el gráfico comparativo en: 'sesion_4/filtrado_convolucion_resultado.png'")
print("\n💡 ANÁLISIS CRÍTICO:")
print("   - Filtro de Media/Gaussiano: Al calcular promedios ponderados, el valor 255 (sal) o 0 (pimienta)")
print("     se mezcla matemáticamente con los vecinos, creando manchas borrosas grises.")
print("   - Filtro de Mediana: Ordena los 49 valores del kernel 7x7. Al ser el ruido un valor extremo (0 o 255),")
print("     queda en las puntas del arreglo y la Mediana escoge siempre un valor intermedio real,")
print("     eliminando el ruido al 100% sin deformar ni difuminar los bordes.")
print("=" * 65)
