import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 65)
print("EJECUTANDO LABORATORIO: CLASIFICADOR DE FORMAS (SESIÓN 6)")
print("=" * 65)

# 1. Generar una imagen sintética a color con varias figuras/monedas de diferentes tamaños
imagen_color = np.zeros((400, 600, 3), dtype=np.uint8) + 30  # Fondo gris oscuro

# Objetos "grandes" (monedas grandes / herramientas)
cv2.circle(imagen_color, (120, 120), 50, (220, 220, 220), -1)
cv2.rectangle(imagen_color, (350, 70), (520, 200), (200, 200, 200), -1)

# Objetos "pequeños" (monedas pequeñas / tuercas)
cv2.circle(imagen_color, (100, 300), 22, (240, 240, 240), -1)
cv2.circle(imagen_color, (250, 250), 18, (240, 240, 240), -1)
cv2.rectangle(imagen_color, (400, 280), (450, 330), (210, 210, 210), -1)

# 2. Pipeline de Procesamiento de Imagen
# Paso A: Escala de grises
gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# Paso B: Umbralización binaria
_, binaria = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)

# Paso C: Limpieza morfológica (Apertura)
kernel = np.ones((3, 3), np.uint8)
binaria_limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

# 3. Detección de Contornos
contornos, _ = cv2.findContours(binaria_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nSe detectaron {len(contornos)} objetos en total.\n")

# Copia para dibujar resultados
resultado = imagen_color.copy()
UMBRAL_AREA_GRANDE = 3500  # Criterio de clasificación

for i, cnt in enumerate(contornos, start=1):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    # Calcular centroide mediante momentos
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2

    # Lógica de clasificación empresarial por área
    if area >= UMBRAL_AREA_GRANDE:
        color_box = (255, 0, 0)  # Azul en BGR (Objeto Grande)
        tipo = "GRANDE"
    else:
        color_box = (0, 0, 255)  # Rojo en BGR (Objeto Pequeño)
        tipo = "PEQUEÑO"

    # Dibujar Bounding Box y Centroide Punto Rojo
    cv2.rectangle(resultado, (x, y), (x + w, y + h), color_box, 2)
    cv2.circle(resultado, (cx, cy), 4, (0, 0, 255), -1)

    print(f"Objeto #{i}: Área = {area:6.1f} px² | Clasificación: {tipo:7s} | Bounding Box: [x={x}, y={y}, w={w}, h={h}]")

# 4. Guardar imagen resultante utilizando matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("1. Imagen Binaria Limpia")
plt.imshow(binaria_limpia, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("2. Clasificador de Objetos (Azul=Grande, Rojo=Pequeño)")
# Convertir de BGR (OpenCV) a RGB (Matplotlib)
plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.savefig('clasificador_formas_resultado.png')

print("\n✅ Proceso completado exitosamente.")
print("📷 Visualización guardada en: 'sesion_6/clasificador_formas_resultado.png'")
print("=" * 65)
