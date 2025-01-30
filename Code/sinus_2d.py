import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress


# Paramètres
d = 0.05 
coef_c = -0.1
c = np.array([coef_c, coef_c])  
dx = 0.05  
dy = 0.05  

# Condition de stabilité
gamma = 0.5
nu = 0.5
if coef_c != 0:
    delta_t2 = nu * dx / np.abs(coef_c)
else:
    delta_t2 = np.inf  # Éviter la division par zéro

delta_t1 = gamma * (dx**2) / (2*d)
dt = min(delta_t1, delta_t2)

# Définition des bornes
Nx = 2  
Ny = 2  
Nt = 2  

# Calcul du nombre de points 
nbrx = int(Nx / dx) + 1  
nbry = int(Ny / dy) + 1  
nbrt = int(Nt / dt) + 1  

# Discrétisation des axes
x = np.linspace(0, Nx, nbrx)  
y = np.linspace(0, Ny, nbry)  
t = np.linspace(0, Nt, nbrt)  

# Initialisation des solutions
solutionsApprochées = np.zeros((nbrx, nbry, nbrt))
solutionExacte = np.zeros((nbrx, nbry, nbrt))

# Fonction solution exacte
def u(x, y, t):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + t)

# Calcul du gradient
def nabla(x, y, t):
    nabla_x = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * (1 + t)
    nabla_y = np.pi * np.cos(np.pi * y) * np.sin(np.pi * x) * (1 + t)
    return np.array([nabla_x, nabla_y])

# Fonction source
def f(x, y, t):
    grad = nabla(x, y, t)
    convection = np.dot(c, grad)
    diffusion = 2 * d * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * (1 + t)
    return np.sin(np.pi * x) * np.sin(np.pi * y) + convection + diffusion

# Conditions initiales
for i in range(nbrx):
    for j in range(nbry):
        solutionsApprochées[i, j, 0] = u(x[i], y[j], 0)  # Condition initiale

# Conditions aux bords
for n in range(nbrt):
    solutionsApprochées[0, :, n] = 0
    solutionsApprochées[-1, :, n] = 0
    solutionsApprochées[:, 0, n] = 0
    solutionsApprochées[:, -1, n] = 0

# Résolution avec différences finies
for n in range(nbrt - 1):  
    for i in range(1, nbrx - 1):  
        for j in range(1, nbry - 1):  
            convection_term = c[0] * (solutionsApprochées[i + 1, j, n] - solutionsApprochées[i, j, n]) / dx + \
                              c[1] * (solutionsApprochées[i, j + 1, n] - solutionsApprochées[i, j , n]) / dy
            diffusion_term = d * (
                (solutionsApprochées[i + 1, j, n] - 2 * solutionsApprochées[i, j, n] + solutionsApprochées[i - 1, j, n]) / dx**2 +
                (solutionsApprochées[i, j + 1, n] - 2 * solutionsApprochées[i, j, n] + solutionsApprochées[i, j - 1, n]) / dy**2
            )
            source_term = f(x[i], y[j], t[n])
            solutionsApprochées[i, j, n + 1] = solutionsApprochées[i, j, n] + dt * (diffusion_term - convection_term + source_term)

# Calcul de la solution exacte
for n in range(nbrt):
    for i in range(nbrx):
        for j in range(nbry):
            solutionExacte[i, j, n] = u(x[i], y[j], t[n])

# Tracé des solutions pour un instant donné
plt.figure(figsize=(10, 6))
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, solutionsApprochées[:, :, -1], cmap="inferno", levels=50)
plt.colorbar(label="Valeur de la solution")
plt.title(f"Solution approchée à t={Nt}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.figure(figsize=(10, 6))
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, solutionExacte[:, :, -1], cmap="inferno", levels=50)
plt.colorbar(label="Valeur de la solution")
plt.title(f"Solution exacte à t={Nt}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Graphique 3D de la solution approchée
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X3D, Y3D = np.meshgrid(x, y)
ax.plot_surface(X3D, Y3D, solutionsApprochées[:, :, -1], cmap="inferno")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Solution")
ax.set_title(f"Solution approchée en 3D à t={Nt}")
plt.show()

# Calcul des erreurs
erreur_L2_temps = np.zeros(nbrt)
erreur_Linf_temps = np.zeros(nbrt)
for n in range(nbrt):
    erreur_L2_temps[n] = np.sqrt(dx* dy * np.sum((solutionsApprochées[:,:, n] - solutionExacte[:,:, n])**2))
    erreur_Linf_temps[n] = np.max(np.abs(solutionsApprochées[:,:, n] - solutionExacte[:,:, n]))


# Régression linéaire log-log sur les erreurs
log_temps = abs(np.log(t[1:])) # On exclut t=0 pour éviter log(0)
log_erreurs = abs(np.log(erreur_Linf_temps[1:]))  # Même exclusion pour les erreurs
slope, intercept, _, _, _ = linregress(log_temps, log_erreurs)

# Affichage de la pente et ajustement
plt.figure(figsize=(10, 6))
plt.plot(log_temps, log_erreurs, 'o', label="Erreurs calculées (log-log)")
plt.plot(log_temps, slope * log_temps + intercept, 'r--', label=f"Régression linéaire (pente = {slope:.2f})")
plt.xlabel("log(t)")
plt.ylabel("log(erreur L_inf)")
plt.title("Régression linéaire log-log des erreurs")
plt.legend()
plt.show()

print(f"Pente estimée de la convergence (ordre) : {slope:.2f}")

# Écart absolu à t[-1]
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, np.abs(solutionsApprochées[:, :, -1] - solutionExacte[:, :, -1]), cmap="inferno", levels=50)
plt.colorbar(label="Écart absolu")
plt.title(f"Écart absolu entre la solution approchée et la solution exacte à t={t[-1]:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.show()