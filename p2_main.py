#p2_main
import numpy as np
from solver_inclusion import resolver_inclusion

# Parámetros base
P = 86400.0
omega = 2.0 * np.pi / P
T_A = 10.0
contrastes = [1, 10, 100]

# Estructura para imprimir la tabla
print(f"{'k_I / k_0':<10} | {'A(S1)/T_A':<12} | {'tau(S1) [h]':<12} | {'A(S2)/T_A':<12} | {'tau(S2) [h]':<12}")
print("-" * 65)

for k_ratio in contrastes:
    # 1. Ejecutar simulación (10 periodos)
    tiempos, S1, S2 = resolver_inclusion(k_ratio=k_ratio, h=0.025, dt=600.0, num_periodos=10)
    
    # 2. Filtrar únicamente el último período (del día 9 al 10)
    t_ultimo_periodo = 9 * P
    mascara = tiempos >= t_ultimo_periodo
    t_fit = tiempos[mascara]
    S1_fit = S1[mascara]
    S2_fit = S2[mascara]
    
    # 3. Ajuste por mínimos cuadrados: T(t) = T_bar + a*sin(wt) + b*cos(wt)
    # Construcción de la matriz de diseño M
    M = np.vstack([np.ones_like(t_fit), np.sin(omega * t_fit), np.cos(omega * t_fit)]).T
    
    resultados = []
    for serie in [S1_fit, S2_fit]:
        # Resolver vector de coeficientes [T_bar, a, b]
        coefs, _, _, _ = np.linalg.lstsq(M, serie, rcond=None)
        a_coef, b_coef = coefs[1], coefs[2]
        
        # Cálculo de amplitud A y desfase tau
        A = np.sqrt(a_coef**2 + b_coef**2)
        # modulo 2pi asegura que el desfase sea positivo
        tau_segundos = (np.arctan2(-b_coef, a_coef) % (2 * np.pi)) / omega
        tau_horas = tau_segundos / 3600.0
        
        resultados.extend([A / T_A, tau_horas])
    
    # 4. Imprimir la fila correspondiente de la tabla
    print(f"{k_ratio:<10} | {resultados[0]:<12.4f} | {resultados[1]:<12.4f} | {resultados[2]:<12.4f} | {resultados[3]:<12.4f}")