import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import IPython.display
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML
from matplotlib import cm


# Define the Euler method for solving differential equations
def euler(f, u0, t0, tf, n):
    t = np.linspace(t0, tf, n+1)
    dt = t[1] - t[0]
    u = np.zeros((n+1, len(u0)))
    u[0, :] = u0
    for i in range(n):
        u[i+1, :] = u[i, :] + dt * f(u[i, :], t[i])
    return u, t



"""
ANNA EULER FRA JUPYTER
def euler(f,u0,t0,tf,n):
    t = np.linspace(t0,tf,n)
    u = np.zeros((n,u0.size))
    u[0,:] = u0
    for i in np.arange(n-1):
        u[i+1,:] = u[i,:] + (t[i+1]-t[i])*(f(u[i],t[i]))
    return u,t
"""


# Parameters
k1 = 0.45
p = 1037.4
Cp = 2600
alpha = (k1 / (Cp * p))
print(alpha)
a = 0.05
b = 0.05
m = 50  # Number of points in the x-direction
n = 50  # Number of points in the y-direction

# Grid setup
x = np.linspace(-a, a, m+2)
y = np.linspace(-b, b, n+2)
h = x[1] - x[0]
k = y[1] - y[0]

# Matrices for the Poisson equation in x and y directions
L1 = alpha * (1/h**2) * sp.diags([1, -2, 1], [-1, 0, 1], shape=(m, m))
I1 = sp.eye(m)
L2 = alpha * (1/k**2) * sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
I2 = sp.eye(n)
A = sp.kron(L1, I2) + sp.kron(I1, L2)  # Kronecker product for 2D Poisson equation

# Boundary conditions
Zm_l = np.zeros(m)
Zm_l[0] = -1/(h**2)
Zm_r = np.zeros(m)
Zm_r[-1] = -1/(h**2)
Zn_l = np.zeros(n)
Zn_l[0] = -1/(k**2)
Zn_r = np.zeros(n)
Zn_r[-1] = -1/(k**2)

# Boundary function
def ute(x):
    return 200

ute_vectorized = np.vectorize(ute)

F = sp.kron(ute_vectorized(x[1:-1]), Zn_l) + sp.kron(ute_vectorized(x[1:-1]), Zn_r) + sp.kron(Zm_l, ute_vectorized(y[1:-1])) + sp.kron(Zm_r, ute_vectorized(y[1:-1]))

# Differential equation function
def f(x, t):
    return A @ x - F

# Initial condition and grid for visualization
X, Y = np.meshgrid(x[1:-1], y[1:-1])
u0 = 20 + np.zeros(m * n)

# Solve the equation using the Euler method
u, t = euler(f, u0, 0, 10, 10000)

# Check when the temperature in the middle reaches 60 degrees
for i in range(len(t)):
    temp_middle = u[i, m*n//2]  # Temperature in the middle of the object
    if temp_middle >= 60:
        print(f"Temperature in the middle of the object reaches 60 degrees at time {t[i]}")
        break


# Visualization with matplotlib
fig = plt.figure(figsize=(20, 10))


# 3D plot
ax1 = fig.add_subplot(131, projection="3d")
Z = np.reshape(u[1000, :], (m, n))
surf = ax1.plot_surface(np.transpose(X), np.transpose(Y), Z, cmap=cm.viridis)
ax1.set_title("3D Plot")


# Color plot
ax2 = fig.add_subplot(132)
pos = ax2.imshow(-Z, cmap='RdBu', interpolation='none')
ax2.set_title("Color Plot")
fig.colorbar(pos, ax=ax2)

# Animation
ax3 = fig.add_subplot(133)
ims = []
for i in range(40):
    im = ax3.imshow(-np.reshape(u[100*i, :], (m, n)), cmap='RdBu', animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ax3.set_title("Animation")

plt.tight_layout()
plt.show()





"""
fig, ax2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 15))
Z = np.reshape(u[1000, :], (m, n))
ax2.plot_surface(np.transpose(X), np.transpose(Y), Z, cmap=cm.Blues)
ax2.set_title("3D Plot")
plt.show()

fig, ax = plt.subplots(figsize=(15, 10))
ims = []
for i in range(40):
    im = ax.imshow(-np.reshape(u[100 * i, :], (m, n)), cmap='RdBu', animated=True)
    if i == 0:
        ax.imshow(-np.reshape(u[0, :], (m, n)), cmap='RdBu')  # show an initial one first
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ax.set_title("Animation")
plt.show()
"""
