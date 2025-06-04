import pyvista
from dolfinx import fem, plot
import numpy as np

def visualize(input_mesh,mark_cells):
    p = pyvista.Plotter()
    #Create a marker for each cell (size_local because we are not worry about parallelization at this stage)
    marker = np.zeros(input_mesh.topology.index_map(input_mesh.topology.dim).size_local,dtype = np.int32)
    marker[mark_cells] = 1
    V = fem.functionspace(input_mesh, ("Lagrange", 1,(input_mesh.geometry.dim,)))
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.cell_data["Marker"] = marker
    grid.set_active_scalars("Marker")
    
    p.add_mesh(grid, show_edges=False, cmap = "grey",flip_scalars= True, clim = [0,1])
    p.view_xy()
    p.remove_scalar_bar()
    p.show()