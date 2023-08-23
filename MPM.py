import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

image = cv2.resize(image, (64, 64))
image = image / 255.0

# Set up the simulation domain
dim = 2  # 2D simulation
n_particles = image.shape[0] * image.shape[1]
grid_size = max(image.shape[:2])
dx = 0.1
dt = 0.001

# Initialize Taichi environment
ti.init(arch=ti.gpu)

# Define grid and particles
grid = ti.Vector.field(dim, dtype=ti.f32, shape=(grid_size, grid_size))
particle_mass = 1
particle_volume = 1
particle_grid_pos = ti.Vector.field(dim, dtype=ti.i32, shape=grid_size)

particle_grid_pos = ti.Vector.field(dim, dtype=ti.i32, shape=(grid_size, grid_size))

# Assign initial material properties
density = 1
c = 10
mu = 1
la = 1

# Assign initial boundary conditions
@ti.kernel
def set_boundary():
    for i in ti.ndrange(grid_size):
        for j in ti.static([0, grid_size-1]):
            grid[i, j].fill(0)
            grid[j, i].fill(0)

# Call the boundary condition initialization kernel
set_boundary()

@ti.kernel
def initialize_particles():
    for i, j in ti.ndrange(image.shape[:2]):
        p = i * image.shape[1] + j
        particle_grid_pos[p] = ti.Vector([i, j])
        # Set particle properties based on image data
        particle_mass[p] = image[i, j]  # Example: use the pixel intensity as the particle mass
        # Other properties can be assigned similarly

@ti.kernel
def update_particles(dt: ti.f32):
    for i in range(num_particles):
        # Update particle velocity using constitutive model
        stress = compute_stress(i)
        velocity[i] += dt * stress @ velocity_gradient[i]
        
        # Update particle position based on updated velocity
        position[i] += dt * velocity[i]
        
        # Check for boundary conditions and handle particle collisions
        handle_boundary_conditions(i)
        
# Define grid dimensions and resolution
grid_size = 64
dx = 1.0 / grid_size

# Define grid variables
velocity = ti.Vector.field(2, dtype=ti.f32, shape=(grid_size, grid_size))
mass = ti.field(dtype=ti.f32, shape=(grid_size, grid_size))
force = ti.Vector.field(2, dtype=ti.f32, shape=(grid_size, grid_size))
cell_type = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))

# Map particle properties to grid
@ti.kernel
def map_to_grid():
    for p in range(num_particles):
        # Compute grid cell containing particle
        base = int(pos[p] / dx - 0.5)
        
        # Compute weight and velocity contribution of particle to surrounding grid cells
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j]) - 1
            weight = w(offset, pos[p])
            dpos = pos[p] - (base + offset) * dx
            vel_contrib = dt * weight * (force[p] / mass[p])
            
            # Accumulate particle contributions to grid variables
            velocity[base + offset] += vel_contrib
            mass[base + offset] += weight * mass[p]
            force[base + offset] += weight * force[p]
            cell_type[base + offset] = 1  # Mark cell as active
            
# Compute grid forces
@ti.kernel
def compute_forces():
    for i, j in velocity:
        # Apply boundary conditions
        if cell_type[i, j] == 0:
            velocity[i, j] = ti.Vector([0.0, 0.0])
            mass[i, j] = 0.0
            force[i, j] = ti.Vector([0.0, 0.0])
        else:
            # Compute forces based on velocity and mass
            force[i, j] = mass[i, j] * gravity
            if i < grid_size - 1:
                force[i, j] += mu * (velocity[i + 1, j] - velocity[i, j]) / dx
            if j < grid_size - 1:
                force[i, j] += mu * (velocity[i, j + 1] - velocity[i, j]) / dx
                
# Update grid velocities and positions
@ti.kernel
def update_grid():
    for i, j in velocity:
        if cell_type[i, j] == 1:
            # Update velocity and position based on computed forces
            velocity[i, j] += dt * force[i, j] / mass[i, j]
            velocity[i, j] *= ti.exp(-dt * damping)
            velocity[i, j] += dt * wind
            pos[i, j] += dt * velocity[i, j]

@ti.kernel
def update_particle_properties(particle_velocity: ti.template(), grid_velocity: ti.template()):
    for p in range(num_particles):
        # Map grid velocity to particle velocity
        px, py = particle_position[p]
        vx, vy = interpolate_velocity(grid_velocity, px, py)
        particle_velocity[p] = ti.Vector([vx, vy])

@ti.kernel
def update_grid_properties(particle_mass: ti.template(), grid_velocity: ti.template(), grid_strain: ti.template()):
    for i, j in grid_velocity:
        # Map particle velocity to grid velocity
        vx_sum = 0.0
        vy_sum = 0.0
        count = 0
        for p in range(num_particles):
            px, py = particle_position[p]
            if is_inside_grid(i, j, px, py):
                vx_sum += particle_mass[p] * particle_velocity[p][0]
                vy_sum += particle_mass[p] * particle_velocity[p][1]
                count += 1
        if count > 0:
            grid_velocity[i, j] = ti.Vector([vx_sum / count, vy_sum / count])
        
        # Map particle strain to grid strain
        strain_sum = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for p in range(num_particles):
            px, py = particle_position[p]
            if is_inside_grid(i, j, px, py):
                dx, dy = px - i, py - j
                strain_sum += particle_mass[p] * ti.Matrix([[dx * dx, dx * dy], [dx * dy, dy * dy]])
        grid_strain[i, j] = strain_sum / grid_cell_area

# Define simulation parameters
num_particles = 10000
dt = 0.01
num_steps = 100

# Initialize simulation
x = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
m = ti.field(dtype=ti.f32, shape=num_particles)
F = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(64, 64))
grid_m = ti.field(dtype=ti.f32, shape=(64, 64))

# Set initial conditions
@ti.kernel
def initialize():
    for i in range(num_particles):
        x[i] = ti.Vector([ti.random(), ti.random()])
        v[i] = ti.Vector([0.0, 0.0])
        m[i] = 1.0

# Define particle update kernel
@ti.kernel
def particle_update():
    for i in range(num_particles):
        F[i] = ti.Vector([0.0, -9.8])  # Example force, gravity
        v[i] += dt * F[i] / m[i]
        x[i] += dt * v[i]
# Initialize simulation
initialize_particles()

# Lists to store particle positions over time
positions_x = []
positions_y = []

# Simulation loop
for step in range(num_steps):
    # Update particle positions
    particle_update()

    # Append current positions to the lists
    positions_x.append(particle_grid_pos.to_numpy()[:, 0])
    positions_y.append(particle_grid_pos.to_numpy()[:, 1])

# Convert lists to NumPy arrays
positions_x = np.array(positions_x)
positions_y = np.array(positions_y)

# Plot particle positions
plt.figure()
for i in range(n_particles):
    plt.plot(positions_x[:, i], positions_y[:, i])

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Particle Positions')
plt.show()
