import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import time

class CreateMicrostructure:
    def __init__(self, dimensions, gridSize, seed, density):
        self.dimensions = dimensions # 2 x 1 array
        self.gridSize = gridSize # 2 x 1 array 
        self.seed = seed
        self.density = density

    def  generateEllipsoidalMicrostructure(self, n, size_bounds, ellipticity_bounds, rotation_bounds):
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
    

    def  generateFiberMicrostructure(self, n, thickness, theta):
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
                    print("I am here")
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

    def rasterizeFibers(self, y_min=0.0, y_max=None):
        if y_max is None:
            y_max = self.dimensions[1]

        dx = self.dimensions[0] / self.gridSize[0]
        dy = self.dimensions[1] / self.gridSize[1]

        # Create a raster grid
        raster = np.zeros((self.gridSize[0], self.gridSize[1]), dtype=int)

        # Create physical coordinate grid (center of each pixel)
        x_lin = (np.arange(self.gridSize[0]) + 0.5) * dx
        y_lin = (np.arange(self.gridSize[1]) + 0.5) * dy
        X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')  # shape (Nx, Ny)

        tan_theta = np.tan(abs(self.theta))
        mask_y = (Y >= y_min) & (Y <= y_max)

        for beta1, beta2 in self.fibers:
            if self.theta > 0:
                # Two boundary lines: x = tan(theta) * y + beta1 and beta2
                x_left = -tan_theta * Y + beta1
                x_right = -tan_theta * Y + beta2
            else:
                # Two boundary lines: x = tan(theta) * y + beta1 and beta2
                x_left = tan_theta * Y + beta1
                x_right = tan_theta * Y + beta2

            # Handle both increasing and decreasing x (depending on theta sign)
            if beta1 < beta2:
                inside = (X >= x_left) & (X <= x_right)
            else:
                inside = (X >= x_right) & (X <= x_left)

            inside &= mask_y  # Restrict to layer region
            raster = np.logical_or(raster, inside)

        return raster.astype(int)

    

# Record the start time
start_time = time.time()

micro = CreateMicrostructure(dimensions=(1.0, 1.0), gridSize=(100, 100), seed=42, density=10)
# micro.generateEllipsoidalMicrostructure(
#     n=100,
#     size_bounds=(0.01, 0.05),
#     ellipticity_bounds=(0.1, 0.2),
#     rotation_bounds=(0, 2 * np.pi)
# )
# # raster = micro.rasterizeEllipsoids()
# raster = micro.rasterizeEllipsoidsVectorized()

micro.generateFiberMicrostructure(n=5, thickness = 0.1, theta = np.radians(30))
raster1 = micro.rasterizeFibers(0,0.5)

micro.generateFiberMicrostructure(n=5, thickness = 0.1, theta = np.radians(-30))
raster2 = micro.rasterizeFibers(0.5,1)

# Combine layers
composite_raster = np.maximum(raster1, raster2)

# Record the end time
end_time = time.time()

# Calculate the total run time
total_run_time = end_time - start_time

# Print the total run time
print(f"Total run time: {total_run_time} seconds")

# Visualize the rasterized grid
# plt.imshow(raster.T, extent=(0, 1, 0, 1), origin='lower', cmap='Greys')
plt.imshow(composite_raster.T, extent=(0, 1, 0, 1), origin='lower', cmap='Greys')
plt.colorbar(label='Inside Ellipsoid')
plt.show()
