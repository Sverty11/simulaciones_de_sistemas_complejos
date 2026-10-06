#p2_main
import numpy as np
from p2_calculo_principal import resolver_inclusion

# Parámetros base
P = 86400.0
omega = 2.0 * np.pi / P
T_A = 10.0
contrastes = [1, 10, 100]
k_0 = 2.3
rho = 1500
c = 1600

############################## 2.1 #############################
omega = 2*np.pi/P
delta_0 = np.sqrt((2*k_0)/(rho*c*omega))
A_1 = np.exp(-0.1/delta_0) #amp @ z=0.1

print(f"{30*"#"} Estimaciones de skin depth: 2.1: {30*"#"}")
for factor in contrastes:
    print(f"k_i = {factor}*k_0")
    k_i = factor*k_0
    delta_i = np.sqrt((2*k_i)/(rho*c*omega))
    perdida_placa = np.exp(-0.2/delta_i)
    print(f"la placa genera pérdiad de {perdida_placa:.3f}")
    A_2 = perdida_placa * A_1   #amp remanente tras placa
    if A_2>(1/np.e):
        print(f"Amp. después de placa: {A_2:.3f} skin depth aún no alcanzada")
    elif A_2<(1/np.e):
        print(f"Skin depth se encuentra dentro de la región de inclusión")
        z_final = 0.1 - np.log(1/(np.e*A_1))*delta_i
        print(f"@ z_c = {z_final:.3f}: Amp(z_c)/A(0) = {A_1 * np.exp(-(z_final-0.1)/delta_i):.3f}\n")    
        continue
    #queremos identificar z tq A(z) = A(0)/e
    z_final = 0.3 - np.log(1/(np.e*A_2))*delta_0
    print(f"@ z_c = {z_final:.3f}: Amp(z_c)/A(0) = {A_2 * np.exp(-(z_final-0.3)/delta_0):.3f}\n")

#######################################################################
print(f"{90*"#"}")

# Estructura para imprimir la tabla
print(f"{'k_I / k_0':<10} | {'A(S1)/T_A':<12} | {'tau(S1) [h]':<12} | {'A(S2)/T_A':<12} | {'tau(S2) [h]':<12}")
print("-" * 65)

for k_ratio in contrastes:
    # 1. Ejecutar simulación (10 periodos)
    tiempos, S1, S2 = resolver_inclusion(k_ratio=k_ratio, h=0.025, dt=600.0, num_periodos=10, guardar_xdmf=True)
    
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