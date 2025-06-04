from dolfinx import fem, plot, io, default_scalar_type, geometry
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities
import ufl
import numpy as np

#homogenization of microstructure under basic assumptions
def homogenization(input_mesh, marker, boundary_dofs, mat1, mat2, Q, V, p):
    #input_mesh: the microscopic mesh 
    #marker: an array of index for material 1
    #boundary_dofs: DoFs on the boundary
    #mat1: the material properties for material 1 (e.g. inclusion)
    #mat2: the material properties for material 2 (e.g. background matrix)
    #Q: a 0th-order discrete Galerkin function space for heterogeneous material properties 
    #V: a 1st-order Lagrangian function space for FEM simulation
    #p: a small pertabation value for numerical homogenization
    
    #Setting up heterogeneous material properties 
    E, nu = Function(Q), Function(Q)
    #Set up background material (e.g. material 2)
    E.x.array[:] = np.full_like(E.x.array[:],mat2[0],dtype = default_scalar_type)
    nu.x.array[:] = np.full_like(E.x.array[:],mat2[1],dtype = default_scalar_type)
    E.x.array[marker] = np.full_like(marker, mat1[0],dtype=default_scalar_type)
    nu.x.array[marker] = np.full_like(marker,mat1[1],dtype=default_scalar_type)
    #Setting up elastic tensor
    C = (E/(1-nu**2))*ufl.as_matrix(
        [[1.0,nu,0],[nu,1.0,0],[0,0,.5*(1-nu)]]
        )
    
    #Setting up the virtual unit macroscopic strain that is scaled by the virtual displacement p
    eps_Mxx = (p)* np.array([ [1,0],[0,0]])
    eps_Mxy = (p)*np.array([ [0,.5],[.5,0]])
    eps_Myy = (p)*np.array([ [0,0],[0,1]])
    
    #Define 3 different displacement function for capturing the three types of boundary conditions 
    u_S1 = Function(V, dtype=default_scalar_type)
    u_S2 = Function(V, dtype=default_scalar_type)
    u_S3 = Function(V, dtype=default_scalar_type)
    
    #Define a lambda function for the pure displacement boundary conditions
    u_S1.interpolate(lambda x: [eps_Mxx[0,0]*x[0] + eps_Mxx[0,1]*x[1], eps_Mxx[1,0]*x[0] + eps_Mxx[1,1]*x[1]])
    u_S2.interpolate(lambda x: [eps_Mxy[0,0]*x[0] + eps_Mxy[0,1]*x[1], eps_Mxy[1,0]*x[0] + eps_Mxy[1,1]*x[1]])
    u_S3.interpolate(lambda x: [eps_Myy[0,0]*x[0] + eps_Myy[0,1]*x[1], eps_Myy[1,0]*x[0] + eps_Myy[1,1]*x[1]])


    #Set up the correspodning 3 virtual bcs 
    bc1= dirichletbc(u_S1,boundary_dofs)
    bc2= dirichletbc(u_S2,boundary_dofs)
    bc3= dirichletbc(u_S3,boundary_dofs)
    
    #Set up trial function for displacement 
    u = ufl.TrialFunction(V)
    #Set up test function 
    v = ufl.TestFunction(V)
    #Set up a constant zero body force microscale 
    f = fem.Constant(input_mesh, default_scalar_type((0, 0)))
    #Set up stress
    sigma = ufl.dot(C, epsilon(u))
    #Assemble the bilinear operator in UFL
    a = ufl.inner(sigma,epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx 

    #virtual problem for axial strain in x direction
    problemxx = LinearProblem(a, L, bcs=[bc1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uhxx = problemxx.solve()
    
    #virtual problem for virtual shear strin
    problemxy = LinearProblem(a, L, bcs=[bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uhxy = problemxy.solve()
    
    #virtual problem for axial strain in y direction
    problemyy = LinearProblem(a, L, bcs=[bc3], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uhyy = problemyy.solve()
    
    #Begin homogenization process by computing the volume of the RVE
    vol = fem.assemble_scalar(fem.form(fem.Constant(input_mesh, default_scalar_type(1))*ufl.dx))
    
    #Compute the microscopic stress field for each virtual macroscopic boundary conditions
    stressxx = ufl.dot(C, epsilon(uhxx)) 
    stressyy = ufl.dot(C, epsilon(uhyy))
    stressxy = ufl.dot(C, epsilon(uhxy))
        
    #Assemble Tangent using Voigt notation
    C11 = fem.assemble_scalar(fem.form(stressxx[0]*ufl.dx))
    C21 = fem.assemble_scalar(fem.form(stressxx[1]*ufl.dx))
    C31 = fem.assemble_scalar(fem.form(stressxx[2]*ufl.dx))
    C12 = fem.assemble_scalar(fem.form(stressyy[0]*ufl.dx))
    C22 = fem.assemble_scalar(fem.form(stressyy[1]*ufl.dx))
    C32 = fem.assemble_scalar(fem.form(stressyy[2]*ufl.dx))
    C13 = fem.assemble_scalar(fem.form(stressxy[0]*ufl.dx))
    C23 = fem.assemble_scalar(fem.form(stressxy[1]*ufl.dx))
    C33 = fem.assemble_scalar(fem.form(stressxy[2]*ufl.dx))
    C_macro = (1/(p*vol))*np.array( [[C11,C12,C13],[C21,C22,C23],[C31,C32,C33]])
    return C_macro



#Define strain as symmetric component of displacement gradient
def epsilon(u):
    eps_t = ufl.sym(ufl.grad(u))
    return ufl.as_vector([eps_t[0, 0], eps_t[1, 1], 2 * eps_t[0, 1]]) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

#Get boundary of the RVE with bottom left corner at (0,0) and top right corner at (1,1)
def boundaryMarker(x):
    xBoundary = np.isclose(x[0],0.0) | np.isclose(x[0],1.0)    
    yBoundary = np.isclose(x[1],0.0) | np.isclose(x[1],1.0)    
    return np.logical_or(xBoundary,yBoundary)
    

