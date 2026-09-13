# ==============================================================================
# ASIGNATURA: Inteligencia Artificial II - Sesión 1
# DOCENTE: Amaury Giovanni Méndez Aguirre
# DESCRIPCIÓN: Solución explicada de los Talleres Analíticos 1 y 2
# ==============================================================================

import numpy as np


def taller_analitico_1():
    """
    TALLER ANALÍTICO 1: INDEXACIÓN Y TENSORES
    """
    print("=" * 60)
    print("SOLUCIÓN TALLER ANALÍTICO 1")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # Pregunta 1: En la Matriz A (5x5), valor exacto de A_{2,3} (base 0)
    # y su representación visual.
    # --------------------------------------------------------------------------
    # Representación de la Matriz A (5x5) dada en el documento:
    A = np.array([
        [0, 255, 255, 255, 0],
        [255, 0, 0, 0, 255],
        [255, 0, 128, 0, 255],
        [255, 0, 0, 0, 255],
        [0, 255, 255, 255, 0]
    ])

    fila, columna = 2, 3
    valor = A[fila, columna]

    print(f"1. El valor exacto en A[{fila},{columna}] es: {valor}")
    print("   Justificación visual:")
    print("   - Fila 2 (tercera fila de la matriz).")
    print("   - Columna 3 (cuarta columna de la matriz).")
    print(f"   - El valor {valor} representa el color NEGRO (brillo 0) dentro del")
    print("     rango discreto de 8 bits [0, 255]. Rellena el fondo alrededor")
    print("     del centro que tiene intensidad gris (128).\n")

    # --------------------------------------------------------------------------
    # Pregunta 2: Cálculo de bytes de memoria para Tensor RGB (1080, 1920, 3)
    # --------------------------------------------------------------------------
    alto = 1080
    ancho = 1920
    canales = 3

    # Número total de elementos (píxeles por canal)
    total_elementos = alto * ancho * canales

    # En imágenes estándar sin comprimir (8 bits por canal = 1 byte por valor)
    bytes_totales = total_elementos * 1
    megabytes = bytes_totales / (1024 * 1024)

    print("2. Cálculo de almacenamiento de memoria:")
    print(f"   - Dimensiones: {alto} x {ancho} x {canales}")
    print(f"   - Total de valores individuales (bytes): {bytes_totales:,} bytes")
    print(f"   - Equivale a aproximadamente: {megabytes:.2f} MB en RAM (sin comprimir).")
    print("-" * 60 + "\n")


def taller_analitico_2():
    """
    TALLER ANALÍTICO 2: TRANSFORMACIONES Y REDES NEURONALES
    """
    print("=" * 60)
    print("SOLUCIÓN TALLER ANALÍTICO 2")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # Pregunta 1: Transposición de una Matriz Identidad de 4x4
    # --------------------------------------------------------------------------
    I_4x4 = np.eye(4)
    I_transpuesta = I_4x4.T

    print("1. Transposición de una Matriz Identidad I (4x4):")
    print("   Matriz Identidad:")
    print(I_4x4)
    print("   Resultado I^T igual a I:", np.array_equal(I_4x4, I_transpuesta))
    print("   Explicación geométrica:")
    print("   - La matriz identidad es una matriz simétrica (A = A^T).")
    print("   - Sus elementos distintos de cero están únicamente en la diagonal principal.")
    print("   - Reflejar la matriz sobre su diagonal principal no cambia la posición")
    print("     de ningún elemento.\n")

    # --------------------------------------------------------------------------
    # Pregunta 2: Aplanamiento (Flattening) de imagen RGB (200, 200, 3)
    # --------------------------------------------------------------------------
    alto, ancho, canales = 200, 200, 3
    num_neuronas = alto * ancho * canales

    print("2. Capa de entrada para Red Neuronal (Flattening):")
    print(f"   - Forma de la imagen: ({alto}, {ancho}, {canales})")
    print(f"   - Cálculo: {alto} * {ancho} * {canales} = {num_neuronas:,} elementos.")
    print(f"   - La capa de entrada requerirá exactamente {num_neuronas:,} neuronas")
    print("     para recibir el vector unidimensional (1D).")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    taller_analitico_1()
    taller_analitico_2()