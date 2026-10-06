from p1_calculo_principal import solve_Calor_suelo_simple as resolver_conduccion

# Casos definidos en la Parte 3 del proyecto
casos = [
    {"h": 0.025,  "dt": 600.0, "desc": "Caso Base"},
    {"h": 0.025,  "dt": 300.0, "desc": "dt / 2"},
    {"h": 0.0125, "dt": 600.0, "desc": "h / 2"}
]

# Cabecera de la tabla de resultados
print(f"{'Escenario':<15} | {'h (m)':<8} | {'dt (s)':<8} | {'Tiempo Ejec. (s)':<18} | {'Error L2':<12}")
print("-" * 70)
W = 1.0          # Ancho del dominio (m)
D = 1.5          # Profundidad del dominio (m)

# Propiedades del material
rho = 1500.0     # Densidad (kg/m^3)
c_p = 1600.0     # Calor específico (J/kg K)
k0 = 2.3         # Conductividad térmica (W/m K)

# Parámetros de la condición inicial y de borde
TR = 10.0        # Temperatura superficial de referencia (°C)
B = 1.0          # Amplitud de la perturbación (°C)

# Discretización exigida para la primera iteración
t_final = 86400.0 # Tiempo final (s) correspondientes a 24 horas

for caso in casos:
    h = caso["h"]
    dt = caso["dt"]
    
    # Ejecutamos con guardar_xdmf=False para aislar el tiempo de procesamiento matemático
    tiempo, error = resolver_conduccion(W, D, rho, c_p, k0, TR, B, h, dt, t_final, guardar_xmdf=False, verbose=False)
    
    # Se utiliza notación científica (.4e) para el error como se solicita
    print(f"{caso['desc']:<15} | {h:<8.4f} | {dt:<8.1f} | {tiempo:<18.4f} | {error:<12.4e}")