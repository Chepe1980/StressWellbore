import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import streamlit as st

# Page configuration
st.set_page_config(layout="wide")
st.title("Wellbore Hoop Stress Analysis")
st.markdown("""
This app calculates and visualizes the hoop stress distribution around a wellbore using both analytical (Kirsch equations) 
and finite difference methods. Adjust the parameters below to explore different scenarios.
""")

# Sidebar for input parameters
with st.sidebar:
    st.header("Input Parameters")
    
    # Wellbore geometry
    wellbore_radius = st.number_input("Wellbore Radius (m)", 0.1, 1.0, 0.1, 0.01)
    min_depth = st.number_input("Minimum Depth (m)", 500, 5000, 1000, 100)
    max_depth = st.number_input("Maximum Depth (m)", 1000, 6000, 2000, 100)
    
    # Geomechanical parameters
    overburden_gradient = st.number_input("Overburden Gradient (MPa/m)", 0.01, 0.03, 0.0226, 0.0001)
    pore_pressure_gradient = st.number_input("Pore Pressure Gradient (MPa/m)", 0.005, 0.015, 0.0098, 0.0001)
    Poisson_ratio = st.slider("Poisson's Ratio", 0.1, 0.4, 0.25, 0.01)
    stress_anisotropy = st.slider("Stress Anisotropy (σH/σh)", 1.0, 2.0, 1.5, 0.1)
    
    # Visualization settings
    threshold_percent = st.slider("Stress Concentration Threshold (%)", 30, 90, 50, 5)
    resolution = st.selectbox("Model Resolution", ["Low", "Medium", "High"], index=1)

# Set resolution parameters
if resolution == "Low":
    depth_points = 3
    theta_points = 18
    radial_points = 10
elif resolution == "Medium":
    depth_points = 5
    theta_points = 36
    radial_points = 20
else:  # High
    depth_points = 7
    theta_points = 72
    radial_points = 30

# Kirsch equations for analytical solution
def kirsch_hoop_stress(r, theta, sigma_H, sigma_h, wellbore_radius, Pp):
    term1 = (sigma_H + sigma_h)/2 * (1 + wellbore_radius**2/r**2)
    term2 = (sigma_H - sigma_h)/2 * (1 + 3*wellbore_radius**4/r**4) * np.cos(2*theta)
    term3 = -Pp * wellbore_radius**2/r**2
    return term1 + term2 + term3

# Main calculation function
def calculate_stresses():
    # Create grid
    depth_range = np.linspace(min_depth, max_depth, depth_points)
    theta = np.linspace(0, 2*np.pi, theta_points)
    r = np.linspace(wellbore_radius, 5*wellbore_radius, radial_points)
    
    # 3D grid for finite difference
    R, Theta, Depth = np.meshgrid(r, theta, depth_range, indexing='ij')
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = Depth
    
    # Calculate in-situ stresses
    K = Poisson_ratio / (1 - Poisson_ratio)
    sigma_v = overburden_gradient * Z
    sigma_h = K * sigma_v
    sigma_H = stress_anisotropy * sigma_h
    Pp = pore_pressure_gradient * Z
    
    # Finite difference setup
    dr = r[1] - r[0]
    dtheta = theta[1] - theta[0]
    dz = depth_range[1] - depth_range[0]
    
    # Sparse matrix construction
    num_points = np.prod(R.shape)
    A = csr_matrix((num_points, num_points))
    b = np.zeros(num_points)
    
    def get_index(i, j, k):
        return i * (Theta.shape[1] * Theta.shape[2]) + j * Theta.shape[2] + k
    
    # Build FD system
    for i in range(1, R.shape[0]-1):
        for j in range(R.shape[1]):
            for k in range(R.shape[2]):
                idx = get_index(i, j, k)
                
                # Radial terms
                A[idx, get_index(i+1, j, k)] = 1/dr**2 + 1/(2*R[i,j,k]*dr)
                A[idx, get_index(i-1, j, k)] = 1/dr**2 - 1/(2*R[i,j,k]*dr)
                A[idx, idx] = -2/dr**2 - 2/(R[i,j,k]**2 * dtheta**2) - 2/dz**2
                
                # Theta terms
                A[idx, get_index(i, (j+1)%R.shape[1], k)] = 1/(R[i,j,k]**2 * dtheta**2)
                A[idx, get_index(i, (j-1)%R.shape[1], k)] = 1/(R[i,j,k]**2 * dtheta**2)
                
                # Depth terms
                if k > 0:
                    A[idx, get_index(i, j, k-1)] = 1/dz**2
                if k < R.shape[2]-1:
                    A[idx, get_index(i, j, k+1)] = 1/dz**2
                
                # Source term
                b[idx] = -sigma_H[i,j,k] * (1 + wellbore_radius**2/R[i,j,k]**2)
    
    # Boundary conditions
    for j in range(R.shape[1]):
        for k in range(R.shape[2]):
            idx = get_index(0, j, k)
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = (sigma_H[0,j,k] + sigma_h[0,j,k])/2 * (1 - 2*np.cos(2*Theta[0,j,k])) - Pp[0,j,k]
    
    # Solve system
    hoop_stress_fd = spsolve(A, b).reshape(R.shape)
    
    return X, Y, Z, R, Theta, Depth, hoop_stress_fd, sigma_H, sigma_h, Pp

# Run calculations
X, Y, Z, R, Theta, Depth, hoop_stress_fd, sigma_H, sigma_h, Pp = calculate_stresses()
mid_depth_idx = len(np.linspace(min_depth, max_depth, depth_points)) // 2

# Create plots
fig = plt.figure(figsize=(18, 12))

## 1. 3D Wellbore Stress Distribution
ax1 = fig.add_subplot(231, projection='3d')
wellbore_surface = ax1.plot_surface(
    X[:,:,mid_depth_idx], Y[:,:,mid_depth_idx], Z[:,:,mid_depth_idx], 
    facecolors=cm.jet(hoop_stress_fd[:,:,mid_depth_idx]/hoop_stress_fd.max()),
    rstride=1, cstride=1, alpha=0.8
)
ax1.set_title('3D Wellbore Hoop Stress')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Depth (m)')

## 2. 3D Stress Concentration
ax2 = fig.add_subplot(232, projection='3d')
threshold = hoop_stress_fd.max() * (threshold_percent/100)
mask = hoop_stress_fd > threshold
scatter = ax2.scatter(
    X[mask], Y[mask], Z[mask], 
    c=hoop_stress_fd[mask], 
    cmap='hot', 
    s=30,
    alpha=0.8
)
ax2.set_title(f'3D Stress Concentration (>{threshold_percent}% of max)')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_zlabel('Depth (m)')

# Add wellbore outline
theta_wall = np.linspace(0, 2*np.pi, 50)
x_wall = wellbore_radius * np.cos(theta_wall)
y_wall = wellbore_radius * np.sin(theta_wall)
z_wall = np.linspace(min_depth, max_depth, 50)
X_wall, Z_wall = np.meshgrid(x_wall, z_wall)
Y_wall, _ = np.meshgrid(y_wall, z_wall)
ax2.plot_wireframe(X_wall, Y_wall, Z_wall, color='black', linewidth=0.5, alpha=0.3)

## 3. Circumferential Stress at Wellbore Wall
theta_fine = np.linspace(0, 2*np.pi, 360)
current_depth = np.linspace(min_depth, max_depth, depth_points)[mid_depth_idx]
stress_at_wall = kirsch_hoop_stress(
    wellbore_radius, theta_fine, 
    sigma_H[0,0,mid_depth_idx], sigma_h[0,0,mid_depth_idx], 
    wellbore_radius, Pp[0,0,mid_depth_idx]
)

ax3 = fig.add_subplot(233)
ax3.plot(np.degrees(theta_fine), stress_at_wall, 'b-')
ax3.axhline(
    (sigma_H[0,0,mid_depth_idx] + sigma_h[0,0,mid_depth_idx])/2 - Pp[0,0,mid_depth_idx], 
    color='r', linestyle='--', label='Avg Stress - Pp'
)
ax3.set_title(f'Hoop Stress at Wellbore Wall\n(Depth = {current_depth:.0f}m)')
ax3.set_xlabel('Angle (degrees)')
ax3.set_ylabel('Hoop Stress (MPa)')
ax3.grid(True)
ax3.legend()

## 4. Radial Stress Distribution
r_fine = np.linspace(wellbore_radius, 5*wellbore_radius, 100)
stress_0deg = kirsch_hoop_stress(
    r_fine, 0, sigma_H[0,0,mid_depth_idx], 
    sigma_h[0,0,mid_depth_idx], wellbore_radius, Pp[0,0,mid_depth_idx]
)
stress_90deg = kirsch_hoop_stress(
    r_fine, np.pi/2, sigma_H[0,0,mid_depth_idx], 
    sigma_h[0,0,mid_depth_idx], wellbore_radius, Pp[0,0,mid_depth_idx]
)

ax4 = fig.add_subplot(234)
ax4.plot(r_fine/wellbore_radius, stress_0deg, 'r-', label='θ=0° (σ_H direction)')
ax4.plot(r_fine/wellbore_radius, stress_90deg, 'b-', label='θ=90° (σ_h direction)')
ax4.axhline(sigma_H[0,0,mid_depth_idx], color='r', linestyle=':', label='Far-field σ_H')
ax4.axhline(sigma_h[0,0,mid_depth_idx], color='b', linestyle=':', label='Far-field σ_h')
ax4.set_title('Radial Stress Decay')
ax4.set_xlabel('Normalized Radius (r/r_w)')
ax4.set_ylabel('Hoop Stress (MPa)')
ax4.grid(True)
ax4.legend()

## 5. 2D Polar Contour
R_2D, Theta_2D = np.meshgrid(r_fine, theta_fine)
X_2D = R_2D * np.cos(Theta_2D)
Y_2D = R_2D * np.sin(Theta_2D)
hoop_stress_2D = kirsch_hoop_stress(
    R_2D, Theta_2D, sigma_H[0,0,mid_depth_idx], 
    sigma_h[0,0,mid_depth_idx], wellbore_radius, Pp[0,0,mid_depth_idx]
)

ax5 = fig.add_subplot(235, polar=True)
contour = ax5.contourf(Theta_2D, R_2D, hoop_stress_2D, 20, cmap='jet')
ax5.set_title('2D Polar Stress Distribution')

## 6. Cartesian Cross-Section
ax6 = fig.add_subplot(236)
contour = ax6.contourf(X_2D, Y_2D, hoop_stress_2D, 20, cmap='jet')
ax6.set_title('Cartesian Stress Distribution')
ax6.set_aspect('equal')

# Add colorbars
fig.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax1, label='Hoop Stress (MPa)')
fig.colorbar(scatter, ax=ax2, label='Hoop Stress (MPa)')
fig.colorbar(contour, ax=ax5, label='Hoop Stress (MPa)')
fig.colorbar(contour, ax=ax6, label='Hoop Stress (MPa)')

plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)

# Additional information
st.markdown("""
### Interpretation Guide:
- **3D Wellbore Stress**: Shows stress distribution on wellbore surface
- **Stress Concentration**: Highlights areas exceeding threshold stress
- **Circumferential Plot**: Shows how stress varies around the wellbore
- **Radial Decay**: Shows how stress decreases with distance from wellbore
- **2D Views**: Provide alternative perspectives of stress distribution

Adjust parameters in the sidebar to explore different scenarios.
""")
