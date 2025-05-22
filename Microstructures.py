import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import time

class CreateMicrostructure:
    def __init__(self, dimensions, gridSize, seed, density):
        self.dimensions = dimensions
        self.gridSize = gridSize
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
    
    def rasterizeEllipsoidsVectorized(self):
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
