print("=" * 65)
print("SOLUCIÓN TALLER ANALÍTICO - SESIÓN 12: CONTANDO PARÁMETROS EN MLP")
print("=" * 65)

# Definición de la arquitectura
n_entradas = 3
n_oculta = 4
n_salida = 1

# 1. Pesos entre Capa de Entrada y Capa Oculta
w1_pesos = n_entradas * n_oculta

# 2. Sesgos en la Capa Oculta
b1_sesgos = n_oculta

# 3. Pesos y Sesgos entre Capa Oculta y Capa de Salida
w2_pesos = n_oculta * n_salida
b2_sesgos = n_salida

# 4. Total de parámetros
total_parametros = w1_pesos + b1_sesgos + w2_pesos + b2_sesgos

print(f"1. Pesos Capa Oculta (W1): {n_entradas} entradas x {n_oculta} neuronas = {w1_pesos} pesos")
print(f"2. Sesgos Capa Oculta (b1): {b1_sesgos} sesgos")
print(f"3. Capa de Salida:")
print(f"   - Pesos (W2) : {n_oculta} ocultas x {n_salida} salida = {w2_pesos} pesos")
print(f"   - Sesgo (b2) : {b2_sesgos} sesgo")
print(f"\n4. TOTAL DE PARÁMETROS ENTRENABLES: {total_parametros}")
print("-" * 65)
print("Desglose:")
print(f" - Capa Oculta : {w1_pesos + b1_sesgos} parámetros")
print(f" - Capa Salida : {w2_pesos + b2_sesgos} parámetros")
print("=" * 65)
