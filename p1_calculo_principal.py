"""
Proyecto Integrado 2: Modelo de conducción térmica
Problema 1: Verificación del modelo de conducción
"""
import os
import numpy as np
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import time
import h5py

# ==============================================================================
# 1. PARÁMETROS FÍSICOS Y DE DISCRETIZACIÓN
# ==============================================================================
# Dimensiones geométricas del suelo
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
h = 0.025        # Tamaño de malla (m)
dt = 60.0       # Paso de tiempo (s)
t_final = 86400.0 # Tiempo final (s) correspondientes a 24 horas


def solve_Calor_suelo_simple(W,D,rho, c_p, k0, TR, B, h, dt, t_final, guardar_xmdf=False, verbose=False):
    """
    W          # Ancho del dominio (m)
    D          # Profundidad del dominio (m)

    # Propiedades del material
    rho        # Densidad (kg/m^3)
    c_p        # Calor específico (J/kg K)
    k0         # Conductividad térmica (W/m K)

    # Parámetros de la condición inicial y de borde
    TR         # Temperatura superficial de referencia (°C)
    B          # Amplitud de la perturbación (°C)

    # Discretización exigida para la primera iteración
    h           # Tamaño de malla (m)
    dt          # Paso de tiempo (s)
    t_final     # Tiempo final (s) correspondientes a 24 horas



    """
    # ==============================================================================
    # 2. GENERACIÓN DE LA MALLA Y ESPACIOS
    # ==============================================================================
    # Cantidad de elementos en X (ancho) y en Y (profundidad z)
    nx = int(W / h)
    ny = int(D / h)

    # Malla rectangular. Asignamos x -> x[0] y z (profundidad) -> x[1]
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([W, D])],
        [nx, ny],
        cell_type=mesh.CellType.triangle
    )

    # Espacio de elementos finitos P1 (Lagrange grado 1)
    V = fem.functionspace(msh, ("Lagrange", 1))

    # ==============================================================================
    # 3. CONDICIÓN INICIAL Y SOLUCIÓN EXACTA
    # ==============================================================================
    # Condición inicial: T(x, z, 0) = TR + B * sin(pi * z / 2D)
    def condicion_inicial(x):
        return TR + B * np.sin(np.pi * x[1] / (2.0 * D))

    T_n = fem.Function(V)
    T_n.name = "Temperatura"
    T_n.interpolate(condicion_inicial)

    # Solución analítica para t > 0
    def solucion_exacta(x, t):
        exponente = -(k0 / (rho * c_p)) * (np.pi / (2.0 * D))**2 * t
        return TR + B * np.sin(np.pi * x[1] / (2.0 * D)) * np.exp(exponente)

    # ==============================================================================
    # 4. CONDICIONES DE BORDE
    # ==============================================================================
    # Temperatura fija en la superficie (z=0, que corresponde a x[1]=0)
    frontera_sup = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0))
    dofs_sup = fem.locate_dofs_topological(V, msh.topology.dim - 1, frontera_sup)
    bc = [fem.dirichletbc(ScalarType(TR), dofs_sup, V)]

    # Nota: En los demás bordes grad(T)·n = 0. En la formulación de elementos finitos,
    # la condición de Neumann homogénea (aislamiento térmico) se cumple de manera
    # natural omitiendo las integrales de contorno en la forma débil.

    # ==============================================================================
    # 5. FORMULACIÓN DÉBIL (EULER IMPLÍCITO)
    # ==============================================================================
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # rho * c * (T_new - T_old)/dt = k0 * div(grad(T_new))
    F = rho * c_p * ufl.inner((T - T_n) / dt, v) * ufl.dx \
    + k0 * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))

    # ==============================================================================
    # 6. RESOLUCIÓN LINEAL Y CONFIGURACIÓN DE EXPORTACIÓN
    # ==============================================================================
    # 1. Crear la función para la solución ANTES de pasársela al solver
    T_h = fem.Function(V)

    # 2. Instanciar el solver pasando 'u=T_h' y el prefijo obligatorio 'petsc_options_prefix'
    problem = LinearProblem(
        a, L, 
        bcs=bc, 
        u=T_h, 
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
        petsc_options_prefix="calor_"
    )

    # Archivo XDMF para visualización
    if guardar_xmdf:
        xdmf = io.XDMFFile(msh.comm, os.path.join(os.getcwd(), "outputs", "conduccion_suelo_h={h}.xdmf"), "w")
        xdmf.write_mesh(msh)
        xdmf.write_function(T_n, 0.0)
    # ==============================================================================
    # 7. BUCLE TEMPORAL
    # ==============================================================================
    t = 0.0
    pasos = int(t_final / dt)
    tiempos_guardados = [0.0]
    if verbose:
        print(f"Iniciando simulación térmica: {pasos} pasos...")
    start_time = time.time()

    for paso in range(1, pasos + 1):
        t += dt
        
        # 1. Resolver el sistema para el nuevo paso
        T_h = problem.solve()
        
        # 2. Actualizar la variable para la próxima iteración
        T_n.x.array[:] = T_h.x.array
        
        # 3. Guardar resultados para ParaView (cada 10 pasos y al final)
        if (paso % 10 == 0 or paso == pasos) and guardar_xmdf:
            xdmf.write_function(T_n, t)
            tiempos_guardados.append(t)

    tiempo_ejecucion = time.time() - start_time
    if guardar_xmdf:
        xdmf.close()



    # Abrimos el archivo en modo "append" (a) para no borrar los datos de FEniCSx
    with h5py.File(os.path.join(os.getcwd(), "outputs", f"conduccion_suelo_h={h}.h5"), "a") as f:
        # Si la columna ya existe de una simulación anterior, la borramos
        if "Tiempos_Fisicos" in f:
            del f["Tiempos_Fisicos"]
        # Creamos un nuevo dataset nativo con el arreglo de tiempos
        f.create_dataset("Tiempos_Fisicos", data=np.array(tiempos_guardados))
                        
    # ==============================================================================
    # 8. ANÁLISIS DE ERROR
    # ==============================================================================
    # Interpolar la solución exacta evaluada en t_final para comparar
    T_exacta = fem.Function(V)
    T_exacta.interpolate(lambda x: solucion_exacta(x, t_final))

    # Calcular norma del error L2(Omega): E = ||T_exacta(t_f) - T_h(t_f)||
    error_form = fem.form(ufl.inner(T_exacta - T_h, T_exacta - T_h) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_L2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM))


    if verbose:
        print("\n" + "="*50)
        print("RESUMEN DE RESULTADOS")
        print("="*50)
        print(f"Tamaño de malla (h)     : {h} m")
        print(f"Paso de tiempo (dt)     : {dt} s")
        print(f"Tiempo de ejecución     : {tiempo_ejecucion:.4f} s")
        print(f"Error Norma L2          : {error_L2:.4e}")
        print("="*50)
    return (tiempo_ejecucion, error_L2)