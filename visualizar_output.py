import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN PROFESIONAL DE MATPLOTLIB CON LATEX
# ==============================================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# ==============================================================================
# EXTRACCIÓN DE DATOS RAW DESDE HDF5
# ==============================================================================
archivo_h5 = os.path.join(os.getcwd(), "outputs", "conduccion_suelo.h5")


with h5py.File(archivo_h5, "r") as f:
    # 1. Extraer la geometría de la malla
    # Buscamos el nombre interno de la malla (usualmente "Grid" o el nombre del archivo)
    nodo_malla = list(f["Mesh"].keys())[0]
    coords = f[f"Mesh/{nodo_malla}/geometry"][:]

    # 2. Filtrar los grados de libertad (nodos) en la línea x = 0.5 m
    tolerancia = 1e-4
    dofs_linea = np.where(np.abs(coords[:, 0] - 0.5) < tolerancia)[0]

    # Extraer las profundidades z (índice 1) y ordenarlas de superficie a fondo
    z_unsorted = coords[dofs_linea, 1]
    idx_orden = np.argsort(z_unsorted)
    z_perfil = z_unsorted[idx_orden]

    # 3. Extraer el historial de temperaturas
    tiempos_exactos = f["Tiempos_Fisicos"][:]
    print(f"largo tiempos_exactos: {len(tiempos_exactos)}")
    temp_group = f["Function/Temperatura"]
    pasos_guardados = list(temp_group.keys())   #las keys son el tiempo en segundos
    
    
    # FEniCSx nombra los pasos como "0", "1", "2", etc. Los ordenamos numéricamente
    pasos_guardados.sort(key=int)
    # print(f"pasos_guardados:{pasos_guardados}")
    # Seleccionamos 6 pasos distribuidos de forma uniforme para el gráfico
    indices_plot = np.linspace(0, len(pasos_guardados) - 1, 6, dtype=int)
    
    fig, ax = plt.subplots(figsize=(6, 8))
    colores = plt.cm.viridis(np.linspace(0, 0.9, len(indices_plot)))

    for i, color in zip(indices_plot, colores):
        paso_str = pasos_guardados[i]
        print(f"paso: {paso_str}")
        # El arreglo completo de temperaturas para este paso
        T_total = temp_group[paso_str][:]
        
        # Extraer solo los nodos de x=0.5m y ordenarlos según la profundidad
        T_perfil = T_total[dofs_linea][idx_orden]
        
        # Calcular el tiempo físico correspondiente
        # El paso "0" de HDF5 es t=0. Los siguientes son n * frecuencia * dt
        t_horas = int(paso_str)/3600.0
        
        ax.plot(T_perfil, z_perfil, label=rf"$t = {t_horas:.1f}$ h", color=color, linewidth=2)

# ==============================================================================
# DISEÑO GEOFÍSICO DEL GRÁFICO
# ==============================================================================
# Invertir el eje Y para que la profundidad aumente hacia abajo (estándar físico)
ax.invert_yaxis()

ax.set_xlabel(r"Temperatura $T$ ($^\circ$C)")
ax.set_ylabel(r"Profundidad $z$ (m)")
ax.set_title(r"Evolución del perfil de temperatura $T(z,t)$ a $x=0.5$ m", pad=15)

# Limitar a la geometría de 1.5 m y el rango de la perturbación inicial
ax.set_ylim(1.5, 0.0)
ax.set_xlim(9.8, 11.2)

ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="black")

plt.tight_layout()
plt.savefig("outputs/perfil_temperatura_extraido.pdf", dpi=300, bbox_inches="tight")
print("Extracción completada. Gráfico exportado como 'perfil_temperatura_extraido.pdf'")
