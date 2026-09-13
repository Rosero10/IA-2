import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("EJECUTANDO LABORATORIO FINAL: LIMPIANDO LA VISIÓN (SESIÓN 3)")
print("=" * 60)

# 1. Crear una imagen de prueba (texto/objeto sintético con ruido)
imagen_base = np.zeros((200, 400), dtype=np.uint8)
cv2.putText(imagen_base, "IA-2", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 10)

# Agregar ruido de sal (puntos blancos en fondo) y ruido de pimienta (puntos negros en objeto)
ruido_sal = np.random.choice([0, 255], size=(200, 400), p=[0.97, 0.03]).astype(np.uint8)
ruido_pimienta = np.random.choice([0, 255], size=(200, 400), p=[0.97, 0.03]).astype(np.uint8)

imagen_ruidosa = cv2.bitwise_or(imagen_base, ruido_sal)
imagen_ruidosa = cv2.bitwise_and(imagen_ruidosa, cv2.bitwise_not(ruido_pimienta))

# 2. Binarización con umbral estático
_, imagen_binaria = cv2.threshold(imagen_ruidosa, 127, 255, cv2.THRESH_BINARY)

# 3. Definir Elemento Estructurante (Kernel 3x3)
kernel = np.ones((3, 3), np.uint8)

# 4. Operación de Apertura (Erosión + Dilatación -> Elimina ruido blanco externo)
imagen_apertura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

# 5. Operación de Cierre (Dilatación + Erosión -> Rellena huecos negros internos)
imagen_cierre = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)

# Operación combinada completa (Apertura seguida de Cierre)
imagen_limpia = cv2.morphologyEx(imagen_apertura, cv2.MORPH_CLOSE, kernel)

# 6. Generación del reporte visual comparativo
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("1. Binarizada con Ruido")
plt.imshow(imagen_binaria, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("2. Apertura (Limpia Sal/Fondo)")
plt.imshow(imagen_apertura, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("3. Cierre (Rellena Huecos/Pimienta)")
plt.imshow(imagen_cierre, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("4. Apertura + Cierre (Restauración)")
plt.imshow(imagen_limpia, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('segmentacion_morfologica_resultado.png')
print("\n✅ Proceso completado exitosamente.")
print("📷 Se ha guardado el comparativo visual en: 'sesion_3/segmentacion_morfologica_resultado.png'")
print("=" * 60)