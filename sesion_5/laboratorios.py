import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("EJECUTANDO LABORATORIO: INSPECTOR DE BORDES (SESIÓN 5)")
print("=" * 65)

# 1. Generar imagen sintética con figuras geométricas y texto
imagen_base = np.zeros((400, 600), dtype=np.uint8)

# Rectángulo (bordes verticales y horizontales claros)
cv2.rectangle(imagen_base, (50, 50), (250, 200), 200, -1)
# Círculo (bordes curvados en múltiples direcciones)
cv2.circle(imagen_base, (450, 120), 80, 255, -1)
# Línea diagonal
cv2.line(imagen_base, (50, 300), (550, 380), 180, 8)
# Texto
cv2.putText(imagen_base, "IA-2 SOBEL", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)

# 2. Sobel X y Sobel Y
sobel_x = cv2.Sobel(imagen_base, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen_base, cv2.CV_64F, 0, 1, ksize=3)

sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# 3. Algoritmo Canny con variaciones de umbrales (Histéresis)
canny_permisivo = cv2.Canny(imagen_base, 10, 50)    # Umbrales muy bajos (capta todo, incluso ruido)
canny_optimo = cv2.Canny(imagen_base, 50, 150)     # Umbrales estándar recomendados
canny_estricto = cv2.Canny(imagen_base, 200, 250)   # Umbrales muy altos (pierde detalles)

# 4. Generar reporte comparativo en matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("1. Imagen Original")
plt.imshow(imagen_base, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("2. Sobel X (Bordes Verticales)")
plt.imshow(sobel_x_abs, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("3. Sobel Y (Bordes Horizontales)")
plt.imshow(sobel_y_abs, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("4. Canny Permisivo (10, 50)")
plt.imshow(canny_permisivo, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("5. Canny Óptimo (50, 150)")
plt.imshow(canny_optimo, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("6. Canny Estricto (200, 250)")
plt.imshow(canny_estricto, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('deteccion_bordes_resultado.png')

print("\n✅ Proceso de detección de bordes completado.")
print("📷 Imagen guardada en: 'sesion_5/deteccion_bordes_resultado.png'")
print("\n💡 EXPERIMENTACIÓN DE UMBRALES EN CANNY:")
print("   - Permisivo (10, 50): Capta bordes muy débiles pero tiende a introducir mucho ruido.")
print("   - Óptimo (50, 150): Mantiene los bordes principales continuos, finos (1 px) y limpios.")
print("   - Estricto (200, 250): Solo detecta los cambios de contraste más extremos; elimina")
print("     detalles sutiles y fragmenta las líneas.")
print("=" * 65)
