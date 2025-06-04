from dolfinx import mesh
from dolfinx.mesh import locate_entities


def rasterize_meshes(input_mesh,ind_func,params):
    #input_mesh: a fem mesh for the space to be define on
    #ind_func: an indicator function to indicate which elements would be 
    #params: parameters for the indicator function
    
    return locate_entities(input_mesh,input_mesh.topology.dim,lambda x:ind_func(x,params))
    
    
    