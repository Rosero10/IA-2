# ==============================================================================
# ASIGNATURA: Inteligencia Artificial II - Sesión 1
# DOCENTE: Amaury Giovanni Méndez Aguirre
# DESCRIPCIÓN: Implementación de Laboratorios Prácticos 1 y Final (Convolución)
# ==============================================================================

import numpy as np


def laboratorio_1_transformaciones_afines():
    """
    TALLER DE LABORATORIO 1: TRANSFORMACIONES AFINES
    Simulación de procesamiento para radiografías sobreexpuestas.
    """
    print("=" * 60)
    print("LABORATORIO 1: TRANSFORMACIONES AFINES (BRILLO Y CONTRASTE)")
    print("=" * 60)

    # 1. Crear matriz de prueba 5x5 con valores aleatorios entre 200 y 255
    #    (simulando la radiografía casi completamente blanca/sobreexpuesta)
    np.random.seed(42)  # Semilla fija para reproducibilidad
    matriz_original = np.random.randint(200, 255, size=(5, 5))

    print("1. Matriz de prueba original (Radiografía sobreexpuesta):")
    print(matriz_original)
    print("-" * 40)

    # 2. Aplicar reducción de contraste del 50% (alpha = 0.5)
    #    y disminuir brillo en 50 unidades (beta = -50.0)
    alpha = 0.5   # Escalar de contraste (reduce rango a la mitad)
    beta = -50.0  # Escalar de brillo (desplaza luminosidad hacia abajo)

    # Ecuación de transformación afín: A_nueva = alpha * A + beta
    matriz_procesada_float = alpha * matriz_original + beta

    # 3. Aplicar clipping np.clip(matriz, min, max) y convertir a np.uint8
    #    Esto previene overflow y mantiene el formato estándar de imagen (0 a 255)
    matriz_procesada = np.clip(matriz_procesada_float, 0, 255).astype(np.uint8)

    # 4. Imprimir resultados comparativos
    print("2. Matriz procesada (Reducción 50% contraste y -50 brillo):")
    print(matriz_procesada)
    print("=" * 60 + "\n")


def laboratorio_final_kernel_convolucion():
    """
    TALLER DE LABORATORIO FINAL: PROGRAMANDO UN KERNEL DE CONVOLUCIÓN
    Simulador básico de paso por kernel para detección de características en CNNs.
    """
    print("=" * 60)
    print("LABORATORIO FINAL: SIMULADOR BÁSICO DE CONVOLUCIÓN (KERNEL)")
    print("=" * 60)

    # 1. Crear las matrices según los diagramas provistos en la guía
    #    Sección de Imagen (I) de 3x3
    I = np.array([
        [100, 100, 100],
        [100, 200, 100],
        [100, 100, 100]
    ], dtype=np.float32)

    # Kernel de Realce (K) de 3x3 para detección de bordes/detalles
    K = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    print("Sección de Imagen (I):\n", I)
    print("\nKernel de Realce (K):\n", K)
    print("-" * 40)

    # 2. Operación de Convolución en el parche local:
    # Paso A: Producto Hadamard (Multiplicación elemento a elemento, operador *)
    producto_hadamard = I * K

    # Paso B: Sumar TODOS los elementos de la matriz resultante
    pixel_central_calculado = np.sum(producto_hadamard)

    # Aplicar saturación/clipping para asegurar rango válido [0, 255]
    pixel_central_final = np.clip(pixel_central_calculado, 0, 255).astype(np.uint8)

    # 3. Imprimir el resultado final
    print("Demostración paso a paso:")
    print("a. Matriz resultado de Producto Hadamard (I * K):\n", producto_hadamard)
    print(f"\nb. Suma total de los elementos (np.sum): {pixel_central_calculado}")
    print(f"c. Valor del Píxel Central Final (tras uint8/clip): {pixel_central_final}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    laboratorio_1_transformaciones_afines()
    laboratorio_final_kernel_convolucion()