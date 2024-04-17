# Koden hentet direkte fra 7_1
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm

k = 50
Cp = 7800
p = 460
alpha = 1.5
#k / (Cp * p)
# bestem antall punkter i gitteret i x-retning
m=100

# sett m+2 punkter mellom 0 og 1
# m+2 fordi vi teller ikke randene 0 og 1 - se randbetingelser for hvorfor
x = np.linspace(-5, 5, m+2)

# avstand mellom punktene
h = x[1] - x[0]

# setter opp matrise tilsvarende Poissonligning i x-retning
L1 = alpha * (1/h**2)*sp.diags([1,-2,1], [-1,0,1], shape=(m,m))

# identitetsmatrise i x-koordinatene
I1 = sp.eye(m)

# antall punkter i gitteret i y-retning
n = 100

# sett n+2 punkter mellom 0 og 1
y = np.linspace(-5, 5, n+2)

# avstand mellom punktene
k = y[1] - y[0]

# setter opp matrise tilsvarende Poissonligning i y-retning
L2 = alpha * (1/k**2)*sp.diags([1,-2,1],[-1,0,1],shape=(n,n))

# identitetsmatrise i y-koordinatene
I2 = sp.eye(n)

# sett sammen matrisa med Kroneckerproduktet
A = sp.kron(L1,I2) + sp.kron(I1,L2)

# Lag en vektor (-1/h^2,0,0,0,...) med m elementer
Zm_l = np.zeros(m)
Zm_l[0] = -1/(h**2)

# Lag en vektor (0,0,0,...,0,-1/h^2) med m elementer
Zm_r = np.zeros(m)
Zm_r[-1] = -1/(h**2)

# Lag en vektor (1/k^2,0,0,0,...) med n elementer
Zn_l = np.zeros(n)
Zn_l[0] = -1/(k**2)

# Lag en vektor (0,0,0,...,0,1/k^2) med n elementer
Zn_r = np.zeros(n)
Zn_r[-1] = -1/(k**2)

def f1(x):
    return np.full_like(x, 200)

def f2(x):
    return np.full_like(x, 200)

def f3(y):
    return np.full_like(y, 200)

def f4(y):
    return np.full_like(y, 200)

# Lag en vektor fra randbetingelser
F = sp.kron(f1(x[1:-1]),Zn_l) + sp.kron(f2(x[1:-1]),Zn_r) + sp.kron(Zm_l,f3(y[1:-1])) + sp.kron(Zm_r,f4(y[1:-1]))


# Eulers metode for generell funksjon f
# trenger initialverdi x_0, initialtid a og sluttid b, og antall tidssteg N
def euler(f,x0,a,b,N):
    t = np.linspace(a,b,N)
    x = np.zeros((N,x0.size))
    x[0,:] = x0
    for i in np.arange(N-1):
        x[i+1,:] = x[i,:] + (t[i+1]-t[i])*(f(x[i],t[i]))
    return x,t

# funksjonen som er høyre side av den differentialligningen, her lik Ax - F
def f(x,t):
    return A @ x - F

# setter opp et rutenett, med indexing u(x_i,y_j)
X, Y = np.meshgrid(x[1:-1],y[1:-1], indexing='ij')

# trenger initialverdien u(x,y,0) = y
U0 = Y

# vektorisering av initialverdien
u0 = np.reshape(U0, m*n)

# løs med Eulers metode for tider 0<t<0.5, med N=10.000 steg
# vi trenger mange steg siden vi bruker forlengs Euler!
u, t = euler(f, u0, 0, 10, 10000)


# en 3d-plott
fig = plt.figure(figsize=(20, 10))

# 3D plot
ax1 = fig.add_subplot(131, projection="3d")
Z = np.reshape(u[1000, :], (m, n))
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis)
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
