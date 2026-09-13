# ==============================================================================
# ASIGNATURA: Inteligencia Artificial II - Sesión 2
# DOCENTE: Amaury Giovanni Méndez Aguirre
# DESCRIPCIÓN: Implementación de los Laboratorios 1 y 2 (Color, Grises e Histograma)
# ==============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def laboratorio_1_transformacion_espacios():
    """
    TALLER DE LABORATORIO 1: TRANSFORMACIÓN DE ESPACIOS
    Conversión BGR a escala de grises ponderada manual vs OpenCV.
    """
    print("=" * 60)
    print("LABORATORIO 1: CONVERSIÓN A ESCALA DE GRISES (PRODUCTO PUNTO)")
    print("=" * 60)

    # 1. Píxel BGR amarillo puro (Azul=0, Verde=255, Rojo=255)
    pixel_amarillo_bgr = np.array([0, 255, 255], dtype=np.float32)

    # 2. Pesos de luminancia ajustados al orden BGR:
    # Y = 0.114 * B + 0.587 * G + 0.299 * R
    pesos_bgr = np.array([0.114, 0.587, 0.299], dtype=np.float32)

    # Cálculo del producto punto manual
    gris_calculado = np.dot(pixel_amarillo_bgr, pesos_bgr)

    print("1 y 2. Píxel BGR (Amarillo puro):", pixel_amarillo_bgr)
    print(f"3. Valor en escala de grises calculado manualmente: {gris_calculado:.2f}")
    print(f"   (Valor entero redondeado uint8: {np.uint8(gris_calculado)})\n")

    # 4. Verificación con OpenCV en una imagen
    # Creamos una imagen sintética de prueba si no existe una en disco
    ruta_imagen = "muestra.jpg"
    if not os.path.exists(ruta_imagen):
        # Crear imagen sintética de 100x100 píxeles color amarillo
        img_prueba = np.zeros((100, 100, 3), dtype=np.uint8)
        img_prueba[:, :] = [0, 255, 255]  # Amarillo en BGR
        cv2.imwrite(ruta_imagen, img_prueba)

    imagen_real = cv2.imread(ruta_imagen)
    img_gris_opencv = cv2.cvtColor(imagen_real, cv2.COLOR_BGR2GRAY)

    print("4. Verificación con cv2.cvtColor:")
    print(f"   Valor de gris obtenido por OpenCV: {img_gris_opencv[0, 0]}")
    print("=" * 60 + "\n")


def laboratorio_2_analisis_estadistico_histograma():
    """
    TALLER DE LABORATORIO 2: ANÁLISIS ESTADÍSTICO DE HISTOGRAMAS
    Cálculo de histogramas por canal y graficado superpuesto.
    """
    print("=" * 60)
    print("LABORATORIO 2: ANÁLISIS ESTADÍSTICO Y HISTOGRAMA RGB/BGR")
    print("=" * 60)

    ruta_imagen = "foto.jpg"
    
    # Si la imagen no existe en la carpeta, creamos una de prueba con gradiente
    if not os.path.exists(ruta_imagen):
        alto, ancho = 300, 300
        imagen_demo = np.zeros((alto, ancho, 3), dtype=np.uint8)
        # Llenar canal Rojo y Verde para simular tonalidad cálida
        imagen_demo[:, :, 0] = np.random.randint(0, 100, (alto, ancho))  # Azul bajo
        imagen_demo[:, :, 1] = np.random.randint(50, 180, (alto, ancho)) # Verde medio
        imagen_demo[:, :, 2] = np.random.randint(150, 255, (alto, ancho))# Rojo alto
        cv2.imwrite(ruta_imagen, imagen_demo)
        print(f"[INFO] Se ha generado '{ruta_imagen}' de prueba para el gráfico.")

    # 1. Cargar imagen en formato BGR
    imagen = cv2.imread(ruta_imagen)

    # 2. Separar los 3 canales (B, G, R)
    canal_b, canal_g, canal_r = cv2.split(imagen)

    # 3 y 4. Calcular e imprimir/graficar histogramas
    colores = ('blue', 'green', 'red')
    canales = (canal_b, canal_g, canal_r)

    plt.figure(figsize=(8, 5))
    plt.title("Distribución de Intensidades por Canal (Histograma Superpuesto)")
    plt.xlabel("Valor del Píxel (0-255)")
    plt.ylabel("Frecuencia (Cantidad de Píxeles)")

    promedios = {}
    for canal, color, nombre in zip(canales, colores, ('Azul (B)', 'Verde (G)', 'Rojo (R)')):
        # Calcular histograma usando OpenCV
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f"Canal {nombre}")
        promedios[nombre] = np.mean(canal)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Guardar el gráfico como imagen para ver el resultado en Codespaces
    archivo_grafico = "histograma_resultado.png"
    plt.savefig(archivo_grafico)
    plt.close()

    print(f"Gráfico del histograma guardado exitosamente como '{archivo_grafico}'.")
    
    # 5. Conclusión analítica automatizada
    print("\n5. Conclusión analítica del Histograma:")
    color_dominante = max(promedios, key=promedios.get)
    for canal_nom, prom in promedios.items():
        print(f"   - Promedio de intensidad {canal_nom}: {prom:.2f}")

    print(f"\n   -> Conclusión: El color dominante en la iluminación de la foto es el **{color_dominante}**, ")
    print("      debido a que presenta las mayores frecuencias acumuladas hacia la derecha del histograma.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    laboratorio_1_transformacion_espacios()
    laboratorio_2_analisis_estadistico_histograma()