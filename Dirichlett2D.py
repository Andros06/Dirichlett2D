import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import matplotlib.pyplot as plt

# Antall punkt i x-retning, Legger til 2 pga rand
m = 50
M = m + 2

# Antall punkt i y-retning, Legger til 2 pga rand
n = 50
N = n + 2

# Setter området for x
Xmin = -5
Xmax = 5

# Setter området for y
Ymin = 0
Ymax = 2


# Lager x verdier mellom Xmin og Xmax med M antall punkt
x = np.linspace(Xmin, Xmax , M)

# Finner steglengden i x-retning
h = x[1]-x[0]

# Setter opp matrise tilsvarende Poissonligning i x-retning
L1 = (1/h**2)*sp.diags([1,-2,1],[-1,0,1],shape=(m,m))

# Identitetsmatrise i x-koordinatene
I1 = sp.eye(m)


# Lager y verdier mellom Ymin og Ymax med N antall punkt
y=np.linspace(Ymin, Ymax, N)

# Finner steglengden i y-retning
k = y[1]-y[0]

# Setter opp matrise tilsvarende Poissonligning i y-retning
L2 = (1/k**2)*sp.diags([1,-2,1],[-1,0,1],shape=(n,n))

# Identitetsmatrise i y-koordinatene
I2 = sp.eye(n)


# Bruker kroneckerproduktet for å danne A
A = sp.kron(L1,I2) + sp.kron(I1,L2)


# Legger inn randbetingelser
Zm_l = np.zeros(m)
Zm_l[0] = -1/(h**2)

Zm_r = np.zeros(m)
Zm_r[-1] = -1/(h**2)

Zn_l = np.zeros(n)
Zn_l[0] = -1/(k**2)

Zn_r = np.zeros(n)
Zn_r[-1] = -1/(k**2)


# funksjonen som gir u(x,0)
def f1(x):
    return 0*x

# funksjonen som gir u(x,2)
def f2(x):
    return np.sin(np.pi*x)

# funksjonen som gir u(5,y)
def f3(y):
    return np.sin(2*np.pi*y)

# funksjonen som gir u(-5,y)
def f4(y):
    return np.sin(2*np.pi*y)


# Lagar en vektor fra randbetingelser
F = sp.kron(f1(x[1:-1]),Zn_l) + sp.kron(f2(x[1:-1]),Zn_r) + sp.kron(Zm_l,f3(y[1:-1])) + sp.kron(Zm_r,f4(y[1:-1]))


# Lager eit rutenett med x verdier langs y akse og y verdier langs x akse
X,Y = np.meshgrid(x[1:-1],y[1:-1], indexing='ij')

# Setter inn funksjonen fra oppgåve
def f(x,y):
    return 0*x  

# Funksjonsverdiene i en array
Z = f(X,Y)

# Reshape array til å få en vektor med f(x,y)
G = np.reshape(Z,(m*n))

# Legg samman f(x,y) med randbetingelsar
F = F + G

# Vi bruker transpose for å gi vår vektor riktig
u = lin.spsolve(A,np.transpose(F))

# Reshaper til en vektor
U = np.reshape(u,(m,n))


# Plotting av figur
fig,ax = plt.subplots(subplot_kw ={"projection":"3d"}, figsize=(15,15))
ax.plot_surface(X, Y, U)
plt.show()