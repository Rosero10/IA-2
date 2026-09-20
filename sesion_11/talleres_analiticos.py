import numpy as np

print("=" * 65)
print("SOLUCIÓN TALLER ANALÍTICO - SESIÓN 11: EL PERCEPTRÓN")
print("=" * 65)

# Entradas del cliente
x1_ingresos = 50
x2_deudas = 20

# Parámetros internos de la neurona
w1 = 0.8
w2 = -0.5
b = -10.0

# 1. Combinación lineal (Z)
z = (x1_ingresos * w1) + (x2_deudas * w2) + b

# 2. Función de activación Escalón (Step Function)
salida = 1 if z >= 0 else 0

print(f"\n1. Cálculo de la Combinación Lineal Z:")
print(f"   Z = ({x1_ingresos} * {w1}) + ({x2_deudas} * {w2}) + ({b})")
print(f"   Z = 40 - 10 - 10 = {z:.1f}")

print(f"\n2. Resultado de la Función Escalón f(Z):")
print(f"   Salida = {salida} -> {'APROBADO' if salida == 1 else 'RECHAZADO'}")

print("\n3. Análisis Empresarial del Peso W2 (-0.5):")
print("   - Tiene sentido que W2 sea negativo porque las Deudas (X2) representan un riesgo financiero.")
print("   - Un peso negativo penaliza el valor de Z: a mayor nivel de deuda, menor es el valor de Z,")
print("     lo que reduce la probabilidad de que la neurona apruebe el crédito.")
print("=" * 65)
