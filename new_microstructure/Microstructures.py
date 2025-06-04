import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from rasterize_meshes import rasterize_meshes
from visualize import visualize
import time

from dolfinx import mesh, fem, plot, io, default_scalar_type, geometry
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities
from mpi4py import MPI
import ufl
import numpy as np

class CreateMicrostructure:
    def __init__(self, dimensions, gridSize, seed, density):
        self.dimensions = dimensions
        self.gridSize = gridSize
        self.seed = seed
        self.density = density        

    #Circular microstructure (geometric isotropic material for benchmarking purpose)
    def circularMarker(self,x,R):
        return (x[0] - .5*self.dimensions[0])**2 + (x[1] - .5*self.dimensions[1])**2 <= R**2
    
    def meshCircular(self,mesh,R):
        marker = rasterize_meshes(mesh, self.circularMarker, R)
        return np.array(marker,dtype = np.int32)
    

    #Ellipsoidal microstructure
    def generateEllipsoidalMicrostructure(self, n, size_bounds, ellipticity_bounds, rotation_bounds):
        np.random.seed(self.seed)
        ellipses = []
        centers = []
        min_distance = (size_bounds[0] + size_bounds[1]) / 2  # Minimum distance between centers

        for _ in range(n):
            # Check if the new center is at least min_distance away from all other centers
            while True:                    
                # Randomly generate the center of the ellipse
                x_center = np.random.uniform(0, self.dimensions[0])
                y_center = np.random.uniform(0, self.dimensions[1])

                # If there are existing centers, use cKDTree to check distances
                if centers:
                    tree = cKDTree(np.array(centers))
                    distances, _ = tree.query([x_center, y_center], k=1)
                    if distances < min_distance:
                        continue  # Too close, generate a new center

                # Valid center found
                break
            
            # Generate major and minor axes of the ellipse
            radius = np.random.uniform(size_bounds[0],size_bounds[1])
            ellipticity = np.random.uniform(ellipticity_bounds[0], ellipticity_bounds[1])

            major_axis = radius
            minor_axis = radius * ellipticity

            # Generate a random rotation angle
            rotation_angle = np.random.uniform(rotation_bounds[0], rotation_bounds[1])

            ellipses.append([x_center, y_center, major_axis, minor_axis, rotation_angle])
            centers.append([x_center, y_center])

        # Store the generated ellipses
        self.ellipses = ellipses


    def ellipsoidsMarker(self,x,params):
        x_center, y_center, major_axis, minor_axis, rotation_angle = params
                
        x_rot = np.cos(rotation_angle)*(x[0]-x_center) + np.sin(rotation_angle)*(x[1] - y_center)
        y_rot = -np.sin(rotation_angle)*(x[0] - x_center) + np.cos(rotation_angle)*(x[1] - y_center)
        return ((x_rot/major_axis)**2 + (y_rot/minor_axis)**2) <= 1
    
    def meshEllipsoids(self,mesh):
        #mesh: a FEniCSx mesh
        
        #init empty marker array
        marker = np.array([])
        for params in self.ellipses:
            grain_marker = rasterize_meshes(mesh, self.ellipsoidsMarker, params)
            marker = np.append(marker,grain_marker)
        return np.array(marker,dtype = np.int32)
    
    def rasterizeEllipsoids(self):
        # Create 2D meshgrid of physical coordinates
        x_lin = (np.arange(self.gridSize[0]) + 0.5) * self.dimensions[0] / self.gridSize[0]
        y_lin = (np.arange(self.gridSize[1]) + 0.5) * self.dimensions[1] / self.gridSize[1]
        X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')  # Shape: (Nx, Ny)
        
        raster = np.zeros_like(X, dtype=int)

        for x_center, y_center, major_axis, minor_axis, rotation_angle in self.ellipses:
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)

            # Shift the grid to ellipse center
            X_shifted = X - x_center
            Y_shifted = Y - y_center

            # Rotate grid
            X_rot = cos_theta * X_shifted + sin_theta * Y_shifted
            Y_rot = -sin_theta * X_shifted + cos_theta * Y_shifted

            # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
            inside = ((X_rot / major_axis) ** 2 + (Y_rot / minor_axis) ** 2) <= 1

            raster = np.logical_or(raster, inside)

        self.raster = raster.astype(int)
        return self.raster

    #Fiber Microstructure
    def generateFiberMicrostructure(self, n, thickness, theta):
        fibers = []
        self.theta = theta

        nThick = n*thickness
        cos_theta = np.cos(theta)
        tan_theta = np.tan(abs(theta))


        yMax = self.dimensions[1]

        if (nThick)/self.dimensions[0] > 0.9:
            thickness = (self.dimensions[0] * 0.9) / n

        spacing = (self.dimensions[0] - nThick) / (n-1)

        for i in range(n):
            # Define fiber extents
            # Bottom left corner
            xL = (spacing + thickness) * i

            # Bottom right corner
            xR = xL + thickness 
        
            # Choose pivot and rotate points
            if theta > 0: # pivot is bottom right point (counter clockwise rotation)
                beta1 = xR - (thickness/cos_theta)
                beta2 = xR
                fibers.append([beta1, beta2]) # x-components of x-intercepts, slope (alpha) is just tan(theta), x = alpha *y + beta 
            else:
                beta1 = xL 
                beta2 = xL + (thickness/cos_theta)
                fibers.append([beta1, beta2]) # x-components of x-intercepts, slope (alpha) is just tan(theta), x = alpha *y + beta  
        

        # Adding ghost nodes (only if a tilted fiber is needed)
        for i in range(n):
            if theta > 0: # Add ghost fibers to the right of the microstructure
                # Bottom left corner
                xL = self.dimensions[0] + (spacing + thickness) * (i+0.5)

                # Bottom right corner
                xR = xL + thickness 

                beta1 = xR - (thickness/cos_theta)
                beta2 = xR

                # Direction of each line from the slope
                dirY = 1
                dirX = - tan_theta

                # Need to check if ghost fiber intersects microstructure domain. We can do this by a bounding box check (similar to ray tracing)
                tx1 = (self.dimensions[0] - xL)/dirX
                tx2 = (0 - xL)/dirX

                ty1 = (yMax)
                ty2 = 0

                tmin = max(min(tx1, tx2), min(ty1, ty2))
                tmax = min(max(tx1, tx2), max(ty1, ty2))

                if tmax>tmin and tmin > 0 and tmax > 0:
                    fibers.append([beta1, beta2]) # intersection with microstructure -> add fiber

            elif theta < 0: # Add ghost fibers to the left of the microstructure
                # Bottom left corner
                xL = - self.dimensions[0] + (spacing + thickness) * (i-0.5)

                # Bottom right corner
                xR = xL + thickness

                beta1 = xL 
                beta2 = xL + (thickness/cos_theta)
                fibers.append([beta1, beta2])

                # Direction of each line from the slope
                dirY = 1
                dirX = tan_theta

                # Need to check if ghost fiber intersects microstructure domain. We can do this by a bounding box check (similar to ray tracing)
                tx1 = (self.dimensions[0] - xR)/dirX
                tx2 = (0 - xR)/dirX

                ty1 = (yMax)
                ty2 = 0

                tmin = max(min(tx1, tx2), min(ty1, ty2))
                tmax = min(max(tx1, tx2), max(ty1, ty2))

                if tmax>tmin and tmin > 0 and tmax > 0:
                    fibers.append([beta1, beta2]) # intersection with microstructure -> add fiber
        
        # Store the generated ellipses
        self.fibers = fibers
        
    def fibersMarkers(self,x,params):
        beta1, beta2 = params
        tan_theta = np.tan(abs(self.theta))
        if self.theta > 0:
            # Two boundary lines: x = tan(theta) * y + beta1 and beta2
            x_left = -tan_theta * x[1] + beta1
            x_right = -tan_theta * x[1] + beta2
        else:
            # Two boundary lines: x = tan(theta) * y + beta1 and beta2
            x_left = tan_theta * x[1] + beta1
            x_right = tan_theta * x[1] + beta2
            
        # Handle both increasing and decreasing x (depending on theta sign)
        if beta1 < beta2:
            return np.logical_and(x[0] >= x_left, x[0] <= x_right)
        else:
            return np.logical_and(x[0] >= x_right,x[0] <= x_left)
        
    def meshFibers(self,mesh):
        #mesh: a FEniCSx mesh
    
        #init empty marker array
        marker = np.array([])
        for params in self.fibers:
            fiber_marker = rasterize_meshes(mesh, self.fibersMarkers, params)
            marker = np.append(marker,fiber_marker)
        return np.array(marker,dtype = np.int32)
        
        
# Record the start time
if __name__=="__main__":
    start_time = time.time()

    input_mesh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0., 0.]), np.array([1., 1.])],
                            [600, 600], cell_type=mesh.CellType.quadrilateral)

    mesh_time = time.time()
    #start_time = time.time()
    micro = CreateMicrostructure(dimensions=(1.0, 1.0), gridSize=(600, 600), seed=42, density=10)
    '''
    micro.generateEllipsoidalMicrostructure(
        n=120,
        size_bounds=(0.05, 0.1),
        ellipticity_bounds=(0.1, 0.2),
        rotation_bounds=(0, 2 * np.pi)
    )
    '''
    micro.generateFiberMicrostructure(n=5, thickness = 0.1, theta = np.radians(30))
    # raster = micro.rasterizeEllipsoids()
    #print(micro.ellipses)
    #raster = micro.rasterizeEllipsoids()

    mark_cells = micro.meshFibers(input_mesh)
    #visualize(input_mesh,np.array(mark_cells,dtype = np.int32))
    print(mark_cells.shape)
    gen_time = time.time()


    # Calculate the total run time
    mesh_run_time = mesh_time - start_time
    gen_run_time = gen_time - mesh_time

    # Print the total run time
    print(f"Mesh run time: {mesh_run_time} seconds")
    print(f"Generation run time: {gen_run_time} seconds")

    # Visualize the rasterized grid
    tdim = input_mesh.topology.dim
    input_mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = mesh.exterior_facet_indices(input_mesh.topology)
    visualize(input_mesh,mark_cells)


    #plt.imshow(raster.T, extent=(0, 1, 0, 1), origin='lower', cmap='Greys')
    #plt.colorbar(label='Inside Ellipsoid')
    #plt.show()
