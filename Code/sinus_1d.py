import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Paramètres
d = 0.01  
c = -0.5 
dx = 0.05  

# Condition de stabilité
gamma = 0.5
nu = 0.5
if c != 0:
    delta_t2 = nu * dx / np.abs(c)
else:
    delta_t2 = np.inf  

delta_t1 = gamma * (dx**2) / d
dt = min(delta_t1, delta_t2)/1.2

# Domaines spatial et temporel
Nx = 2  
Nt = 5
nbrx = int(Nx / dx) + 1 
nbrt = int(Nt / dt) + 1  
x = np.linspace(0, Nx, nbrx)  
t = np.linspace(0, Nt, nbrt)  

# Initialisation des solutions
solutionsApprochées = np.zeros((nbrx, nbrt))
solutionExacte = np.zeros((nbrx, nbrt))

# Fonction solution exacte
def u(x, t):
    return np.sin(np.pi * x) * (1 + t)

def f(x, t):
    return (np.pi**2 * (1 + t) * d * np.sin(np.pi * x) +
            np.pi * (1 + t) * c * np.cos(np.pi * x) +
            np.sin(np.pi * x))

# Conditions initiales
for i in range(nbrx):
    solutionsApprochées[i, 0] = u(x[i], 0)  # Condition initiale en t=0

for j in range(nbrt):
    solutionsApprochées[0, j] = 0  
    solutionsApprochées[-1, j] = 0  

# Résolution avec différences finies
for n in range(nbrt - 1):  
    for i in range(1, nbrx - 1):  
        if c <= 0:
            solutionsApprochées[i, n + 1] = (
                solutionsApprochées[i, n]
                + dt * (
                    d * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n]) / dx**2
                    - c * (solutionsApprochées[i + 1, n] - solutionsApprochées[i, n]) / dx
                    + f(x[i], t[n])
                )
            )
        else:
            solutionsApprochées[i, n + 1] = (
                solutionsApprochées[i, n]
                + dt * (
                    d * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n]) / dx**2
                    - c * (solutionsApprochées[i, n] - solutionsApprochées[i - 1, n]) / dx
                    + f(x[i], t[n])
                )
            )

# Calcul de la solution exacte
for n in range(nbrt):
    for i in range(nbrx):
        solutionExacte[i, n] = u(x[i], t[n])

# Tracé des solutions pour tous les instants
plt.figure(figsize=(12, 8))
for n in range(nbrt):
    plt.clf()  
    plt.plot(x, solutionExacte[:, n], label="Solution exacte", linewidth=2, color='orange')
    plt.plot(x, solutionsApprochées[:, n], label="Solution approchée", linestyle="--", linewidth=2, color='green')
    plt.title(f"Solutions exacte et approchée à t={t[n]:.2f}")
    plt.xlabel("Espace (x)")
    plt.ylabel("Valeur de la solution")
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)

plt.show()


# Calcul de l'écart absolu point par point
ecart_absolu = np.abs(solutionsApprochées - solutionExacte)  
plt.figure(figsize=(10, 6))
plt.imshow(ecart_absolu, extent=[0, Nt, 0, Nx], origin="lower", aspect="auto", cmap="seismic")
plt.colorbar(label="Écart absolu")
plt.title("Écart absolu entre la solution approchée et la solution exacte")
plt.xlabel("Temps (t)")
plt.ylabel("Espace (x)")
plt.show()


###############################################################################################################################################
#On refait les calculs pour obtenir l'erreur en norme infinie en fonction de différents dt

# Initialisation des erreurs
errors_Linf = []
dt_values = [dt / 1.2, dt / 2, dt / 3]

# Résolution pour chaque dt
for dt in dt_values:
    # Calcul du nombre de points temporels en fonction de dt
    nbrt = int(Nt / dt) + 1  # Nombre de points en temps
    t = np.linspace(0, Nt, nbrt)  # Vecteur temporel

    # Initialisation des solutions
    solutionsApprochées = np.zeros((nbrx, nbrt))
    solutionExacte = np.zeros((nbrx, nbrt))

    # Conditions initiales
    for i in range(nbrx):
        solutionsApprochées[i, 0] = u(x[i], 0)  # Condition initiale en t=0

    # Conditions aux limites
    for j in range(nbrt):
        solutionsApprochées[0, j] = 0  # Condition aux bords en x=0
        solutionsApprochées[-1, j] = 0  # Condition aux bords en x=Nx

    # Résolution avec différences finies
    for n in range(nbrt - 1):
        for i in range(1, nbrx - 1):
            if c <= 0:
                solutionsApprochées[i, n + 1] = (
                    solutionsApprochées[i, n]
                    + dt * (
                        d * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n]) / dx**2
                        - c * (solutionsApprochées[i + 1, n] - solutionsApprochées[i, n]) / dx
                        + f(x[i], t[n])
                    )
                )
            else:
                solutionsApprochées[i, n + 1] = (
                    solutionsApprochées[i, n]
                    + dt * (
                        d * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n]) / dx**2
                        - c * (solutionsApprochées[i, n] - solutionsApprochées[i - 1, n]) / dx
                        + f(x[i], t[n])
                    )
                )

    # Calcul de la solution exacte
    for n in range(nbrt):
        for i in range(nbrx):
            solutionExacte[i, n] = u(x[i], t[n])

    # Calcul de l'erreur en norme infinie pour ce dt
    erreur_Linf = np.max(np.abs(solutionsApprochées - solutionExacte))
    errors_Linf.append(erreur_Linf)

# Tracé des erreurs L∞ en fonction de dt (en log-log)
plt.figure(figsize=(10, 6))
plt.loglog(dt_values, errors_Linf, label="Erreur L∞", marker='o', color='b')
plt.title("Erreur L∞ en fonction de dt (log-log)")
plt.xlabel("Pas temporel Δt")
plt.ylabel("Erreur L∞")
plt.grid(True)

# Régression linéaire (log-log)
log_dt = np.log(dt_values)
log_errors = np.log(errors_Linf)
slope, intercept, _, _, _ = linregress(log_dt, log_errors)

# Tracé de la droite de régression
plt.plot(dt_values, np.exp(intercept) * np.array(dt_values) ** slope, 'r--', label=f"Régression linéaire (pente = {slope:.2f})")

plt.legend()
plt.show()