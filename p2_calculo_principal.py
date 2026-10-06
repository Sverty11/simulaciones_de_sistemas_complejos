#p2_calculo_principal

import numpy as np
import ufl
import gmsh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, geometry
from dolfinx.io import gmshio
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

def generar_malla_conforme(h, W, D, x, y, w, h):
    """
    x, y, w, h : coordenadas de inclusión, proporcionales a las coordenadas W, D, partiendo en (0,0) en la ezquina 
    superior izquierda
    """
    """Genera una malla 2D con una inclusión rectangular asegurando conformidad."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("suelo_inclusion")
    
    # Geometría: Dominio total y rectángulo de la inclusión
    suelo = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, W, D)
    inclusion = gmsh.model.occ.addRectangle(x, y, 0.0, w, h)
    
    # Fragmentar para asegurar que la malla respete la frontera interior
    gmsh.model.occ.fragment([(2, suelo)], [(2, inclusion)])
    gmsh.model.occ.synchronize()
    
    # Identificar y asignar Physical Groups
    superficies = gmsh.model.getEntities(2)
    for dim, tag in superficies:
        centro = gmsh.model.occ.getCenterOfMass(dim, tag)
        # Si el centro de masa está dentro de los límites de la inclusión
        if x < centro[0] < x+w and y < centro[1] < y+h:
            gmsh.model.addPhysicalGroup(2, [tag], 2, name="Inclusion")
        else:
            gmsh.model.addPhysicalGroup(2, [tag], 1, name="Suelo")
            
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(2)
    
    # Importar directamente a dolfinx
    msh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return msh, cell_tags

def resolver_inclusion(W,D,coords_inclusion,rho, c_p, k0, TR, B, h, dt, P, num_periodos, k_ratio, guardar_xmdf=False, verbose=False):
    # (k_ratio, h=0.025, dt=600.0, num_periodos=10):
    """
    Resuelve el problema térmico con inclusión y retorna las series de tiempo
    para los puntos de control S1 y S2.
    """
    # Parámetros físicos
    TA = 10.0,
    omega = 2.0 * np.np.pi / P
    t_final = num_periodos * P
    
    # 1. Malla Conforme y Espacios
    x_inc, y_inc, w_inc, h_inc = coords_inclusion
    msh, cell_tags = generar_malla_conforme(h, W, D, x_inc, y_inc, w_inc, h_inc)
    V = fem.functionspace(msh, ("Lagrange", 1)) 
    
    # 2. Definición de la Conductividad Térmica Heterogénea (k) usando DG-0
    Q_k = fem.functionspace(msh, ("DG", 0))
    k_field = fem.Function(Q_k)
    celdas_suelo = cell_tags.find(1)
    celdas_inc = cell_tags.find(2)
    k_field.x.array[celdas_suelo] = k0
    k_field.x.array[celdas_inc] = k0 * k_ratio

    # 3. Condición Inicial
    T_n = fem.Function(V)
    T_n.interpolate(lambda x: np.full_like(x[0], TR))

    # 4. Condición de Borde Dinámica en la superficie (z=0 -> y=0)
    frontera_sup = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
    dofs_sup = fem.locate_dofs_topological(V, 1, frontera_sup)
    
    T_bc = fem.Function(V)
    T_bc.interpolate(lambda x: np.full_like(x[0], TR)) # t=0
    bc = fem.dirichletbc(T_bc, dofs_sup)

    # 5. Formulación Débil
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # a(T, v) = \int rho*c/dt T*v dx + \int k * grad(T)*grad(v) dx
    a_form = fem.form(rho * c_p / dt * ufl.inner(T, v) * ufl.dx + k_field * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx)
    # L(v) = \int rho*c/dt T_n * v dx
    L_form = fem.form(rho * c_p / dt * ufl.inner(T_n, v) * ufl.dx)

    # Pre-ensamblar la matriz (es constante en el tiempo)
    A = assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    from petsc4py import PETSc
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("jacobi")

    T_h = fem.Function(V)
    b = fem.Function(V)

    # 6. Preparar puntos de monitoreo S1(0.5, 0.4) y S2(0.1, 0.4)
    puntos = np.array([[0.5, 0.4, 0.0], [0.1, 0.4, 0.0]])
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    celdas_colision = geometry.compute_collisions_points(bb_tree, puntos)
    celdas = geometry.compute_colliding_cells(msh, celdas_colision, puntos)
    
    historial_t = []
    historial_S1 = []
    historial_S2 = []

    # 7. Bucle Temporal
    t = 0.0
    pasos = int(t_final / dt)
    
    for paso in range(1, pasos + 1):
        t += dt
        
        # Actualizar valor de la condición de borde en t
        T_bc.interpolate(lambda x: np.full_like(x[0], TR + TA * np.sin(omega * t)))
        
        # Ensamblar vector lado derecho y aplicar lifting para la BC
        with b.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b.x.petsc_vec, L_form)
        apply_lifting(b.x.petsc_vec, [a_form], [[bc]])
        b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b.x.petsc_vec, [bc])

        # Resolver
        solver.solve(b.x.petsc_vec, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        
        T_n.x.array[:] = T_h.x.array[:]
        
        # Registrar temperaturas (ejecución serial simplificada)
        if len(celdas.links(0)) > 0 and len(celdas.links(1)) > 0:
            val_S1 = T_h.eval(puntos[0:1], [celdas.links(0)[0]])[0]
            val_S2 = T_h.eval(puntos[1:2], [celdas.links(1)[0]])[0]
            historial_t.append(t)
            historial_S1.append(val_S1)
            historial_S2.append(val_S2)

    return np.array(historial_t), np.array(historial_S1), np.array(historial_S2)