import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import matplotlib.pyplot as plt

# bestem antall punkter i gitteret i x-retning
m=20

# Setter grenser til x
Xmin = -5
Xmax = 5

# Setter grenser til y
Ymin = 0
Ymax = 2

# sett m+2 punkter mellom Xmin og Xmax
# m+2 fordi vi teller ikke randene Xmin og Xmax - se randbetingelser for hvorfor
x=np.linspace(Xmin, Xmax ,m+2)

# avstand mellom punktene
h=x[1]-x[0]

# setter opp matrise tilsvarende Poissonligning i x-retning
L1 = (1/h**2)*sp.diags([1,-2,1],[-1,0,1],shape=(m,m))

# identitetsmatrise i x-koordinatene
I1 = sp.eye(m)

# antall punkter i gitteret i y-retning
n=20

# sett n+2 punkter mellom 0 og 1
y=np.linspace(Ymin, Ymax, n+2)

# avstand mellom punktene
k = y[1]-y[0]

# setter opp matrise tilsvarende Poissonligning i y-retning
L2 = (1/k**2)*sp.diags([1,-2,1],[-1,0,1],shape=(n,n))

# identitetsmatrise i y-koordinatene
I2 = sp.eye(n)

# sett sammen matrisa med Kroneckerproduktet
A = sp.kron(L1,I2) + sp.kron(I1,L2)

# Hvis vi ønsker kan vi skrive ut matrisa med:
print(A.toarray())





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

# Lag en vektor fra randbetingelser
F = sp.kron(f1(x[1:-1]),Zn_l) + sp.kron(f2(x[1:-1]),Zn_r) + sp.kron(Zm_l,f3(y[1:-1])) + sp.kron(Zm_r,f4(y[1:-1]))


# Vi har antatt at funksjonen f(x,y) i Poissonligning er lik null. Hvis ikke må vi lage en vektor f(x,y)

# lag et rutenett fra punktene i x og y
# velger indexing='ij' siden vi bruker F[i,j] = F[x_i,y_j], ikke F[x_j,y_i]
X,Y = np.meshgrid(x[1:-1],y[1:-1], indexing='ij')

def f(x,y):
    return 0*x # sett din funksjon inn her 

# funksjonsverdiene i en array
Z = f(X,Y)

# reshape array til å få en vektor med f(x,y)
G = np.reshape(Z,(m*n))

# legg sammen f(x,y) med randbetingelsene 
F = F + G



# Vi bruker transpose for å gi vår vektor riktig
u = lin.spsolve(A,np.transpose(F))

# reshaper til en vektor
U = np.reshape(u,(m,n))

# lag figuren
fig,ax = plt.subplots(subplot_kw ={"projection":"3d"}, figsize=(15,15))

# plotter
ax.plot_surface(X, Y, U)

plt.show()