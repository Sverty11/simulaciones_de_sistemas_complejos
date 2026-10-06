import os
import numpy as np
import ufl
import gmsh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, geometry
import gmsh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import numpy as np
import gmsh
import meshio
from mpi4py import MPI
from dolfinx import io, mesh
import numpy as np
import gmsh
import meshio
from mpi4py import MPI
from dolfinx import io, mesh

def generar_malla_conforme(h, W=1.0, D=1.5):
    """Genera la malla con Gmsh, convierte con meshio y carga con FEniCSx."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        gmsh.model.add("suelo_inclusion")
        
        # 1. Construcción CAD
        suelo = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, W, D)
        inclusion = gmsh.model.occ.addRectangle(0.4, 0.1, 0.0, 0.2, 0.2)
        
        # 2. Fragmentar para asegurar conformidad
        gmsh.model.occ.fragment([(2, suelo)], [(2, inclusion)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        
        # 3. Etiquetado Físico
        superficies = gmsh.model.getEntities(2)
        for dim, tag in superficies:
            centro = gmsh.model.occ.getCenterOfMass(dim, tag)
            # Usamos un margen holgado para las coordenadas del centro de masa
            if 0.39 < centro[0] < 0.61 and 0.09 < centro[1] < 0.31:
                gmsh.model.addPhysicalGroup(2, [tag], 2, name="Inclusion")
            else:
                gmsh.model.addPhysicalGroup(2, [tag], 1, name="Suelo")
                
        # 4. Mallado
        point_tags = gmsh.model.getEntities(0)
        gmsh.model.mesh.setSize(point_tags, h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        
        gmsh.model.mesh.generate(2)
        
        # 5. Guardar a disco en lugar de usar gmshio
        gmsh.write("malla_temp.msh")
        
    finally:
        gmsh.finalize()
        
    # 6. Conversión con Meshio a formato XDMF
    malla_msh = meshio.read("malla_temp.msh")
    
    # Extraer puntos y celdas (forzando 2D)
    puntos = malla_msh.points[:, :2] 
    celdas_tri = malla_msh.cells_dict["triangle"]
    etiquetas = malla_msh.cell_data_dict["gmsh:physical"]["triangle"]
    
    # Guardar la geometría base
    malla_base = meshio.Mesh(points=puntos, cells=[("triangle", celdas_tri)])
    meshio.write("malla_geometria.xdmf", malla_base)
    
    # Guardar los subdominios físicos (Cell Tags)
    malla_tags = meshio.Mesh(
        points=puntos, 
        cells=[("triangle", celdas_tri)], 
        cell_data={"Subdominios": [etiquetas]}
    )
    meshio.write("malla_tags.xdmf", malla_tags)
    
    # 7. Leer de vuelta de forma nativa con FEniCSx (como en nsbenchx)
    with io.XDMFFile(MPI.COMM_WORLD, "malla_geometria.xdmf", "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")
        
    # Asegurar que el mapa de conectividad exista antes de cargar los tags
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim)
    
    with io.XDMFFile(MPI.COMM_WORLD, "malla_tags.xdmf", "r") as xdmf:
        cell_tags = xdmf.read_meshtags(msh, name="Grid")
        
    return msh, cell_tags

def resolver_inclusion(k_ratio, h=0.025, dt=600.0, num_periodos=10, guardar_xdmf=False, verbose=False):
    """
    Resuelve el problema térmico con inclusión y retorna las series de tiempo
    para los puntos de control S1 y S2.
    """
    # Parámetros físicos
    rho, c_p, k0 = 1500.0, 1600.0, 2.3
    TR, TA = 10.0, 10.0
    P = 86400.0
    omega = 2.0 * np.pi / P
    t_final = num_periodos * P
    
    # 1. Malla Conforme y Espacios
    msh, cell_tags = generar_malla_conforme(h)
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

    # ==========================================================================
    # CONFIGURACIÓN DEL ARCHIVO XDMF
    # ==========================================================================
    if guardar_xdmf:
        ruta_guardado = os.path.join(os.getcwd(), "outputs", f"p2_2_k={k_ratio}.xdmf")
        xdmf = io.XDMFFile(msh.comm, ruta_guardado, "w")
        xdmf.write_mesh(msh)
        xdmf.write_function(T_n, 0.0)

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

        if guardar_xdmf and (paso % 10 == 0 or paso == pasos):  #guardamos cada 10 pasos
            xdmf.write_function(T_n, t)

        # Registrar temperaturas (ejecución serial simplificada)
        if len(celdas.links(0)) > 0 and len(celdas.links(1)) > 0:
            val_S1 = T_h.eval(puntos[0:1], [celdas.links(0)[0]])[0]
            val_S2 = T_h.eval(puntos[1:2], [celdas.links(1)[0]])[0]
            historial_t.append(t)
            historial_S1.append(val_S1)
            historial_S2.append(val_S2)
        
    if guardar_xdmf:
        xdmf.close()
        if verbose:
            print(f"xmdf guardado @ {ruta_guardado}")
    return np.array(historial_t), np.array(historial_S1), np.array(historial_S2)