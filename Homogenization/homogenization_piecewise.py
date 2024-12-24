#Import all the important files from FENICSX
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type, geometry
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities
from mpi4py import MPI
import ufl
import numpy as np

class microstructure:

    def __init__(self,inputLength,inputNumOfElementsPerSide):
        # the length of the square
        self.L = inputLength
        # the number of elements along one side of the square
        self.NePerSide = inputNumOfElementsPerSide
        self.mesh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-self.L, -self.L]), np.array([self.L, self.L])],
                         [self.NePerSide, self.NePerSide], cell_type=mesh.CellType.quadrilateral)
        
        self.setStiff = False
        self.setSoft = False
        self.setVoid = False
        
    def setStiffMaterial(self,inputEStiff,inputNuStiff):
        self.E_Stiff = inputEStiff
        self.nuStiff = inputNuStiff
        self.setStiff = True
        
    def setSoftMaterial(self,inputEStiff,inputNuStiff):
        self.E_Soft = inputEStiff
        self.nuSoft = inputNuStiff
        self.setSoft = True
        
    def setVoidRatio(self,inputStiffVoid,inputSoftVoid):
        self.stiffVoid = inputStiffVoid
        self.softVoid = inputSoftVoid
        self.setVoid = True
        
    
    def setGeomParam(self,inputX1,inputX2,inputR1, inputR2, inputp1,inputp2,inputTheta):
        self.X1 = inputX1
        self.X2 = inputX2
        self.R1 = inputR1
        self.R2 = inputR2
        self.p1 = inputp1
        self.p2 = inputp2
        self.theta = inputTheta
        self.setParam = True
        
    def setGenetic(self,input):
        self.vars = input
         
    def cellStiff(self,x):
        #return entries inside the hard core
        #return np.logical_and(np.abs(x[0]) <= self.coreL ,np.abs(x[1]) <= self.coreL)
        xPrime = (x[0]-self.X1)*np.cos(self.theta) + (x[1]-self.X2)*np.sin(self.theta)
        yPrime = (x[0]-self.X1)*np.sin(self.theta) - (x[1]-self.X2)*np.cos(self.theta)
        return np.abs(xPrime / (self.R1))**self.p1 +np.abs(yPrime/(self.R2))**self.p2 <= 1     
        
        
    #Not used    
    def cellSoft(self,x):
        #return entries in the soft ring
        return np.logical_or(np.abs(x[0]) >= self.coreL ,np.abs(x[1]) >= self.coreL)
    
    #Define strain as symmetric component of displacement gradient
    
    def epsilon(self,u):
        eps_t = ufl.sym(ufl.grad(u))
        return ufl.as_vector([eps_t[0, 0], eps_t[1, 1], 2 * eps_t[0, 1]]) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    #Define stress
    #Might need to change this to Voigt Notation if anistropy occur 
    #Correction: Only if we are modeling the microstructure as an anistropic material (or else we just input material parameter)
    def sigma(self,u):
        C = (self.E/(1-self.nu**2))*ufl.as_matrix(
            [[1.0,self.nu,0],[self.nu,1.0,0],[0,0,.5*(1-self.nu)]]
        )
        sigv = ufl.dot(C, self.epsilon(u))
        return  sigv

     
    def generateRVE(self):
        #Now define a discontinous lagrange constant space
        Q = fem.functionspace(self.mesh,("DG",0))
        self.E, self.nu  = Function(Q) , Function(Q)
        cells = np.arange(self.NePerSide**2)
        """
        #Get the cell locations
        cells_stiff = locate_entities(self.mesh,self.mesh.topology.dim,lambda x:self.cellStiff(x))
        #cells_soft = locate_entities(self.mesh,self.mesh.topology.dim,lambda x:self.cellSoft(x))
        cells_soft = np.arange(self.NePerSide**2)
        cells_soft = np.setdiff1d(cells_soft,cells_stiff)
        num_stiff = cells_stiff.size
        num_soft = cells_soft.size
        num_stiff_void = int(np.ceil(self.stiffVoid*num_stiff))
        num_soft_void = int(np.ceil(self.softVoid*num_soft))
        #print(cells_stiff)
        #print(cells_soft)
        #print(num_stiff)
        #print(num_soft)        
        
        #Set up a random number generator
        rng = np.random.default_rng(12345)
        cells_stiff_void_idx = rng.choice(num_stiff,num_stiff_void)
        cells_stiff_void = cells_stiff[cells_stiff_void_idx]
        cells_stiff = np.setdiff1d(cells_stiff,cells_stiff_void)
        
        cells_soft_void_idx = rng.choice(num_soft,num_soft_void)
        cells_soft_void = cells_soft[cells_soft_void_idx]
        cells_soft = np.setdiff1d(cells_soft,cells_soft_void)
        """
        cells_stiff = cells[self.vars == 1]
        cells_soft = cells[self.vars == 2]
        
        #Assign material properties for lambda_
        self.E.x.array[cells_stiff] = np.full_like(cells_stiff,self.E_Stiff,dtype=default_scalar_type)
        self.E.x.array[cells_soft] = np.full_like(cells_soft,self.E_Soft,dtype=default_scalar_type)
        #self.E.x.array[cells_stiff_void] = np.full_like(cells_stiff_void,np.finfo(np.float64).eps,dtype=default_scalar_type)
        #self.E.x.array[cells_soft_void] = np.full_like(cells_soft_void,np.finfo(np.float64).eps,dtype=default_scalar_type)
        
        #Assign material properties for mu
        self.nu.x.array[cells_stiff] = np.full_like(cells_stiff,self.nuStiff,dtype=default_scalar_type)
        self.nu.x.array[cells_soft] = np.full_like(cells_soft,self.nuSoft,dtype=default_scalar_type)
        #self.nu.x.array[cells_stiff_void] = np.full_like(cells_stiff_void,np.finfo(np.float64).eps,dtype=default_scalar_type)
        #self.nu.x.array[cells_soft_void] = np.full_like(cells_soft_void,np.finfo(np.float64).eps,dtype=default_scalar_type)
        
        #Visualization purpose only 
        self.cells_stiff = cells_stiff
        self.cells_soft = cells_soft
        #self.cells_stiff_void = cells_stiff_void
        #self.cells_soft_void = cells_soft_void
        """
        p = pyvista.Plotter()
        #Create a marker for each cell (size_local because we are not worry about parallelization at this stage)
        marker = np.zeros(self.mesh.topology.index_map(self.mesh.topology.dim).size_local,dtype = np.int32)
        marker[cells_stiff] = 1
        marker[cells_soft] = 2
        marker[cells_stiff_void] = 0
        marker[cells_soft_void] = 0        
        V = fem.functionspace(self.mesh, ("Lagrange", 1,(self.mesh.geometry.dim,)))
        topology, cell_types, geometry = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.cell_data["Marker"] = marker
        grid.set_active_scalars("Marker")
        p.add_mesh(grid, show_edges=False)
        p.view_xy()
        p.show()
        """
        
    def visualizeMicrostructure(self):
        p = pyvista.Plotter()
        #Create a marker for each cell (size_local because we are not worry about parallelization at this stage)
        marker = np.zeros(self.mesh.topology.index_map(self.mesh.topology.dim).size_local,dtype = np.int32)
        marker[self.cells_stiff] = 1
        marker[self.cells_soft] = 2
        #marker[self.cells_stiff_void] = 0
        #marker[self.cells_soft_void] = 0        
        V = fem.functionspace(self.mesh, ("Lagrange", 1,(self.mesh.geometry.dim,)))
        topology, cell_types, geometry = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.cell_data["Marker"] = marker
        grid.set_active_scalars("Marker")
        p.add_mesh(grid, show_edges=False)
        p.view_xy()
        p.show()
        
        
    def boundary(self,x):
        return np.isclose(np.abs(x[0]),self.L) | np.isclose(np.abs(x[1]),self.L)     
    
    
    def getHomogenizedProperties(self):
        #Define a linear Lagrange function space
        V = fem.functionspace(self.mesh, ("Lagrange", 1,(self.mesh.geometry.dim,)))
        #Define the 3 boundary loading
        eps_Mxx = (1e-8)* np.array([ [1,0],[0,0]])
        eps_Mxy = (1e-8)*np.array([ [0,1],[1,0]])
        eps_Myy = (1e-8)*np.array([ [0,0],[0,1]])
    
        
        fdim = self.mesh.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(self.mesh, fdim, self.boundary)
        dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=boundary_facets)
        
        #Define 3 different displacement function for capturing the three types of boundary conditions 
        u_S1 = Function(V, dtype=default_scalar_type)
        u_S2 = Function(V, dtype=default_scalar_type)
        u_S3 = Function(V, dtype=default_scalar_type)

        #Define a lambda function for the pure displacement boundary conditions
        u_S1.interpolate(lambda x: [eps_Mxx[0,0]*x[0] + eps_Mxx[0,1]*x[1], eps_Mxx[1,0]*x[0] + eps_Mxx[1,1]*x[1]])
        u_S2.interpolate(lambda x: [eps_Mxy[0,0]*x[0] + eps_Mxy[0,1]*x[1], eps_Mxy[1,0]*x[0] + eps_Mxy[1,1]*x[1]])
        u_S3.interpolate(lambda x: [eps_Myy[0,0]*x[0] + eps_Myy[0,1]*x[1], eps_Myy[1,0]*x[0] + eps_Myy[1,1]*x[1]])
        
        #Set up the correspodning 3 bcs 
        bc1= dirichletbc(u_S1,dofs)
        bc2= dirichletbc(u_S2,dofs)
        bc3= dirichletbc(u_S3,dofs)
        
        """
        #Debug case
        eps_Mxx1 = (1e-2)* np.array([ [1,0],[0,0]])
        u_Debug1 = Function(V, dtype=default_scalar_type)
        u_Debug1.interpolate(lambda x: [eps_Mxx1[0,0]*x[0] + eps_Mxx1[0,1]*x[1], eps_Mxx1[1,0]*x[0] + eps_Mxx1[1,1]*x[1]])
        bc_t1 = dirichletbc(u_Debug1,dofs)
        
        eps_Mxx2 = (1e-3)* np.array([ [1,0],[0,0]])
        u_Debug2 = Function(V, dtype=default_scalar_type)
        u_Debug2.interpolate(lambda x: [eps_Mxx2[0,0]*x[0] + eps_Mxx2[0,1]*x[1], eps_Mxx2[1,0]*x[0] + eps_Mxx2[1,1]*x[1]])
        bc_t2 = dirichletbc(u_Debug2,dofs)
        
        eps_Mxx3 = (1e-4)* np.array([ [1,0],[0,0]])
        u_Debug3 = Function(V, dtype=default_scalar_type)
        u_Debug3.interpolate(lambda x: [eps_Mxx3[0,0]*x[0] + eps_Mxx3[0,1]*x[1], eps_Mxx3[1,0]*x[0] + eps_Mxx3[1,1]*x[1]])
        bc_t3 = dirichletbc(u_Debug3,dofs)
        """
        
        #Set up trial function for displacement 
        u = ufl.TrialFunction(V)
        #Set up test function 
        v = ufl.TestFunction(V)
        #Set up a constant zero body force microscale 
        f = fem.Constant(self.mesh, default_scalar_type((0, 0)))
        #Assemble the bilinear operator in UFL
        a = ufl.inner(self.sigma(u), self.epsilon(v)) * ufl.dx
        L = ufl.dot(f, v) * ufl.dx 
        
        problemxx = LinearProblem(a, L, bcs=[bc1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uhxx = problemxx.solve()

        problemxy = LinearProblem(a, L, bcs=[bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uhxy = problemxy.solve()

        problemyy = LinearProblem(a, L, bcs=[bc3], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uhyy = problemyy.solve()
        
        """
        #More Debug
        print("Start Though experiment")
        problem1 = LinearProblem(a, L, bcs=[bc_t1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_t1 = problem1.solve()
        s1 = self.sigma(u_t1) 
        A_11 = fem.assemble_scalar(fem.form(s1[0]*ufl.dx))
        A_21 = fem.assemble_scalar(fem.form(s1[1]*ufl.dx))
        A_31 = fem.assemble_scalar(fem.form(s1[2]*ufl.dx))
        print(A_11,A_21,A_31)
        
        problem2 = LinearProblem(a, L, bcs=[bc_t2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_t2 = problem2.solve() 
        s2 = self.sigma(u_t2) 
        B_11 = fem.assemble_scalar(fem.form(s2[0]*ufl.dx))
        B_21 = fem.assemble_scalar(fem.form(s2[1]*ufl.dx))
        B_31 = fem.assemble_scalar(fem.form(s2[2]*ufl.dx))
        print(B_11,B_21,B_31)
        
        problem3 = LinearProblem(a, L, bcs=[bc_t3], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_t3 = problem3.solve()
        s3 = self.sigma(u_t3) 
        D_11 = fem.assemble_scalar(fem.form(s3[0]*ufl.dx))
        D_21 = fem.assemble_scalar(fem.form(s3[1]*ufl.dx))
        D_31 = fem.assemble_scalar(fem.form(s3[2]*ufl.dx))
        print(D_11,D_21,D_31)
        print("end of experiment")
        """
        
        """
        p = pyvista.Plotter()
        topology, cell_types, geometry = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        # Attach vector values to grid and warp grid by vector
        vals = np.zeros((geometry.shape[0], 3))
        #Change this line for between uhxx, uhyy, and uhxy for visualization
        vals[:, :len(uhxx)] = uhxx.x.array.reshape((geometry.shape[0], len(uhxx)))
        grid["u"] = vals
        #print(grid["u"])
        actor_0 = p.add_mesh(grid, style="wireframe", color="k")
        warped = grid.warp_by_vector("u", factor=100)
        actor_1 = p.add_mesh(warped, opacity=0.8)
        p.view_xy()
        p.show_axes()
        p.show()
        """
        #Begin homogenization process
        unitTestDummy = fem.Constant(self.mesh, default_scalar_type(1))
        vol = fem.assemble_scalar(fem.form(unitTestDummy*ufl.dx))
        
        stressxx = self.sigma(uhxx) 
        stressyy = self.sigma(uhyy)
        stressxy = self.sigma(uhxy)
        
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
        C_macro = (1/(1e-8*vol))*np.array( [[C11,C12,C13],[C21,C22,C23],[C31,C32,C33]])
        return C_macro