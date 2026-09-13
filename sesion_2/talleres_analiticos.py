# ==============================================================================
# ASIGNATURA: Inteligencia Artificial II - Sesión 2
# DOCENTE: Amaury Giovanni Méndez Aguirre
# DESCRIPCIÓN: Solución explicada del Taller Analítico 1 (Slicing y Rendimiento)
# ==============================================================================

import numpy as np


def taller_analitico_1():
    """
    TALLER ANALÍTICO 1: OPERACIONES CON TENSORES Y SLICING
    """
    print("=" * 60)
    print("SOLUCIÓN TALLER ANALÍTICO 1 - SESIÓN 2")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # Pregunta 1: Dimensiones (shape) e información de:
    # recorte = imagen[100:200, 300:400, 1] en una imagen de (1080, 1920, 3)
    # --------------------------------------------------------------------------
    # Simulación de la imagen con la forma (1080, 1920, 3) -> (Alto, Ancho, Canales)
    imagen_simulada = np.zeros((1080, 1920, 3), dtype=np.uint8)
    recorte = imagen_simulada[100:200, 300:400, 1]

    print("1. Análisis del corte 'recorte = imagen[100:200, 300:400, 1]':")
    print(f"   - Dimensión exactas (recorte.shape): {recorte.shape}")
    print("   - Explicación de las dimensiones:")
    print("     * Filas (Alto): 200 - 100 = 100 píxeles.")
    print("     * Columnas (Ancho): 400 - 300 = 100 píxeles.")
    print("     * Canal: Al seleccionar un índice escalar ': , : , 1', se extrae un solo canal.")
    print("   - Información específica que contiene:")
    print("     Contiene una matriz 2D de intensidades del Canal Verde (índice 1 en BGR/RGB)")
    print("     correspondiente a la región rectangular de la imagen.\n")

    # --------------------------------------------------------------------------
    # Pregunta 2: Eficiencia de Slicing vs Bucle 'for' anidado
    # --------------------------------------------------------------------------
    print("2. Justificación de eficiencia (Slicing vs Bucle for):")
    print("   - Vectorización y C Bajo el Capó:")
    print("     El slicing de NumPy no itera píxel por píxel en Python.")
    print("     Llama internamente a bloques en lenguaje C precompilados en memoria.")
    print("   - Continuidad de Memoria (Stride/Contiguous memory):")
    print("     Slicing accede a regiones de memoria contigua en la RAM mediante saltos")
    print("     de punteros (strides) sin crear copias innecesarias (utiliza 'Views').")
    print("   - Sobrecarga de Python (Python Overhead):")
    print("     Dos bucles 'for' anidados en Python para una imagen Full HD ejecutarían")
    print("     más de 2 millones de iteraciones a nivel de intérprete, lo cual es")
    print("     órdenes de magnitud más lento.")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    taller_analitico_1()