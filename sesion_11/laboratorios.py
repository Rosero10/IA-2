import numpy as np

print("=" * 65)
print("EJECUTANDO LABORATORIO: COMPUERTA LÓGICA OR CON PERCEPTRÓN (SESIÓN 11)")
print("=" * 65)

# 1. Definición de la Función de Activación Escalón
def funcion_escalon(z):
    return 1 if z >= 0 else 0

# 2. Estructura del Perceptrón mediante Producto Punto
def perceptron(X, W, b):
    z = np.dot(X, W) + b
    return funcion_escalon(z)

# Tabla de Verdad de Entradas Binarias
entradas = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

# -------------------------------------------------------------
# RETO: Configuración de Pesos y Sesgo para la Compuerta OR
# -------------------------------------------------------------
pesos_or = np.array([0.5, 0.5])
sesgo_or = -0.2

print(f"\nParámetros configurados para Compuerta OR:")
print(f" -> Vector de Pesos (W) : {pesos_or}")
print(f" -> Sesgo (b)          : {sesgo_or}\n")

print("Evaluación de la Tabla de Verdad para OR:")
print("-" * 45)
print(f"{'Entrada [X1, X2]':<20} | {'Z Calculado':<12} | {'Salida':<8}")
print("-" * 45)

exito = True
salidas_esperadas = [0, 1, 1, 1]

for x, esperada in zip(entradas, salidas_esperadas):
    z = np.dot(x, pesos_or) + sesgo_or
    salida = perceptron(x, pesos_or, sesgo_or)
    print(f"Input: {str(x):<12} | Z = {z:5.1f}      | Result: {salida}")
    if salida != esperada:
        exito = False

print("-" * 45)
if exito:
    print("✅ COMPUERTA OR SOLUCIONADA EXITOSAMENTE.")
else:
    print("❌ Hay errores en las salidas esperadas.")

print("=" * 65)
