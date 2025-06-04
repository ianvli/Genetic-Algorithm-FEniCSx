from Microstructures import CreateMicrostructure
from visualize import visualize
from homogenization import homogenization
import time
import numpy as np

from dolfinx import mesh, fem
from dolfinx.fem import functionspace
from mpi4py import MPI


#Start benchmark scripts
start_time = time.time()
R = 1
Ne = 600
#Create Mesh
input_mesh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0., 0.]), np.array([1., 1.])],
                        [Ne, Ne], cell_type=mesh.CellType.quadrilateral)
mesh_time = time.time()

#Rasterize Mesh
micro = CreateMicrostructure(dimensions=(1.0, 1.0), gridSize=(600, 600), seed=42, density=10)
mark_cells = micro.meshCircular(input_mesh,R)
gen_time = time.time()

#Setting up function space on the mesh 
Q = fem.functionspace(input_mesh,("DG",0))
V = fem.functionspace(input_mesh, ("Lagrange", 1,(input_mesh.geometry.dim,)))
#Get boundary dofs 
tdim = input_mesh.topology.dim
input_mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = mesh.exterior_facet_indices(input_mesh.topology)
boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=boundary_facets)
func_time = time.time()

#homogenization
print("Begin calcuating macro elastic tensor")
p = 1e-1
Al_prop = [68e9,0.32]
Ti_prop = [110e9,0.31]

C_al = (Al_prop[0]/(1-Al_prop[1]**2))*np.array(
        [[1.0,Al_prop[1],0],[Al_prop[1],1.0,0],[0,0,.5*(1-Al_prop[1])]]
        )
print(f"Elastic tensor of aluminium is \n {C_al}")
C_ti = (Ti_prop[0]/(1-Ti_prop[1]**2))*np.array(
        [[1.0,Ti_prop[1],0],[Ti_prop[1],1.0,0],[0,0,.5*(1-Ti_prop[1])]]
        )
print(f"Elastic tensor of Titanium is \n {C_ti}")
print(":")
C_macro = homogenization(input_mesh, mark_cells, boundary_dofs, Ti_prop, Al_prop, Q, V, p)
h_time = time.time()

#Visualize
#visualize(input_mesh,mark_cells)
print(f"The volume fraction of the inclusion is {len(mark_cells)/Ne**2}")
print(C_macro)
print("")

# Calculate the total run time
mesh_run_time = mesh_time - start_time
gen_run_time = gen_time - mesh_time
func_run_time = func_time - gen_time
h_run_time = h_time - func_time

# Print the total run time
print(f"Mesh run time: {mesh_run_time} seconds")
print(f"Generation run time: {gen_run_time} seconds")
print(f"Setting up function space time: {func_run_time} seconds")
print(f"homogenization time: {h_run_time} seconds")
print(f"Total run time: {mesh_run_time + gen_run_time+ func_run_time + h_run_time} seconds")
