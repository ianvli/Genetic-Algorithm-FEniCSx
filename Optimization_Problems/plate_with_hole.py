from mpi4py import MPI
import gmsh
from dolfinx.io.gmshio import model_to_mesh
import ufl
from dolfinx import fem, io
from dolfinx import default_scalar_type
import dolfinx.fem.petsc
from dolfinx import fem, la
import numpy as np

def generate_mesh():
    L, R = 1.0, 0.1
    h = 0.02

    gdim = 2  # domain geometry dimension
    fdim = 1  # facets dimension

    gmsh.initialize()
    #gmsh.option.setNumber("General.Terminal",0)
    occ = gmsh.model.occ
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        plate = occ.addRectangle(0.0, 0.0, 0.0, L, L)  # (x, y, z, dx, dy)
        notch = occ.addDisk(0.0, 0.0, 0.0, R, R)  # (x, y, z, Rx, Ry=Rx)

        # cut "plate" with "notch"
        outDimTags, _ = occ.cut([(gdim, plate)], [(gdim, notch)])
        # get tag of newly created object
        perf_plate = outDimTags[0][1]

        occ.synchronize()

        # identify boundary tags from bounding box
        eps = 1e-3  # tolerance for bounding box
        left = gmsh.model.getEntitiesInBoundingBox(
            -eps, -eps, -eps, eps, L + eps, eps, dim=fdim
        )[0][1]
        bottom = gmsh.model.getEntitiesInBoundingBox(
            -eps, -eps, -eps, L + eps, eps, eps, dim=fdim
        )[0][1]
        right = gmsh.model.getEntitiesInBoundingBox(
            L - eps, -eps, -eps, L + eps, L + eps, eps, dim=fdim
        )[0][1]
        top = gmsh.model.getEntitiesInBoundingBox(
            -eps, L - eps, -eps, L + eps, L + eps, eps, dim=fdim
        )[0][1]
        # tag physical domains and facets
        gmsh.model.addPhysicalGroup(gdim, [perf_plate], 1)
        gmsh.model.addPhysicalGroup(fdim, [bottom], 1)
        gmsh.model.addPhysicalGroup(fdim, [right], 2)
        gmsh.model.addPhysicalGroup(fdim, [top], 3)
        gmsh.model.addPhysicalGroup(fdim, [left], 4)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        gmsh.model.mesh.generate(gdim)

        domain, cells, facets = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        gmsh.finalize()
    return gdim,fdim,domain, cells, facets


def strain(u, repr="vectorial"):
    eps_t = ufl.sym(ufl.grad(u))
    if repr == "vectorial":
        return ufl.as_vector([eps_t[0, 0], eps_t[1, 1], 2 * eps_t[0, 1]])
    elif repr == "tensorial":
        return eps_t
    
def stress(C, u, repr="vectorial"):
    sigv = ufl.dot(C, strain(u))
    if repr == "vectorial":
        return sigv
    elif repr == "tensorial":
        return ufl.as_matrix([[sigv[0], sigv[2]], [sigv[2], sigv[1]]])
    
def solveMacroProblem(C,gdim,fdim,domain,facets,w1,w2,ifSave = False,):
    #gdim,fdim,domain, _ , facets =generate_mesh()
    
    # Define function space
    V = fem.functionspace(domain, ("P", 2, (gdim,)))

    # Define variational problem
    du = ufl.TrialFunction(V)
    u_ = ufl.TestFunction(V)
    u = fem.Function(V, name="Displacement")
    a_form = ufl.inner(stress(C,du), strain(u_)) * ufl.dx

    # uniform traction on top boundary
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)
    T = fem.Constant(domain, 10.0e6)
    n = ufl.FacetNormal(domain)
    L_form = ufl.dot(T * n, u_) * ds(2)

    V_ux, _ = V.sub(0).collapse()
    V_uy, _ = V.sub(1).collapse()
    bottom_dofs = fem.locate_dofs_topological(V.sub(1), fdim, facets.find(1))
    left_dofs = fem.locate_dofs_topological(V.sub(0), fdim, facets.find(4))

    ux0 = fem.Function(V_ux)
    uy0 = fem.Function(V_uy)
    bcs = [
        fem.dirichletbc(default_scalar_type(0), left_dofs, V.sub(0)),
        fem.dirichletbc(default_scalar_type(0), bottom_dofs, V.sub(1)),
    ]


    problem = fem.petsc.LinearProblem(a_form, L_form, u=u, bcs=bcs)
    problem.solve()
    
    V0 = fem.functionspace(domain, ("DG", 0, (3,)))
    sig_exp = fem.Expression(stress(C,u), V0.element.interpolation_points())
    sig = fem.Function(V0, name="Stress")
    sig.interpolate(sig_exp)
    
    strain_energy = ufl.inner(strain(u,repr = "tensorial"),stress(C,u,repr = "tensorial"))*ufl.dx
    strain_energy_vol = fem.assemble_scalar(fem.form(strain_energy))
    #print("The strain energy (J) is:")
    #print(strain_energy_vol)
    
    sigx_max = max(sig.sub(0).collapse().x.array)
    sigx_norm = np.mean(sig.sub(0).collapse().x.array)
    Kt = sigx_max/sigx_norm
    #print("The stress concentration factor is:")
    #print(Kt)
    #print(sig.x.array.shape)
    #print(sig.x.array.shape)
    #L2_sig = fem.assemble_scalar(fem.form(ufl.inner(sig,sig)*ufl.dx))
    #print(L2_sig)
    #print(sig.sub(0).collapse().x.array)
    #print(sig.sub(0).collapse().x.array.shape)
    #print(max(sig.sub(0).collapse().x.array))
    #print(np.mean(sig.sub(0).collapse().x.array))
    norm_inf_sig = la.norm(sig.x, type=dolfinx.la.Norm.linf)
    #print(norm_inf_u)
    #print(np.linalg.norm(sig.x.array,np.inf))
    
    norm_L2_sig = la.norm(sig.x, type=dolfinx.la.Norm.l2)
    #print(norm_L2_u)
    #print(np.linalg.norm(sig.x.array,2))
    
    norm_L1_u = la.norm(sig.x, type=dolfinx.la.Norm.l1)
    #print(norm_L1_u)
    #print(np.linalg.norm(sig.x.array,1))
    
    if ifSave:
        vtk = io.VTKFile(domain.comm, "linear_elasticity.pvd", "w")
        vtk.write_function(u, 0)
        vtk.write_function(sig, 0)
        vtk.close()
        
    return w1*(Kt-1) + w2*strain_energy_vol
