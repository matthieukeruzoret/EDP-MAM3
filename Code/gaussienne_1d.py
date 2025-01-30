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
Nx = 4  
Nt = 1  
nbrx = int(Nx / dx) + 1
nbrt = int(Nt / dt) + 1
x = np.linspace(-2, 2, nbrx)  # Espace centré sur [-2, 2]
t = np.linspace(0, Nt, nbrt)

# Initialisation des solutions
solutionsApprochées = np.zeros((nbrx, nbrt))
solutionExacte = np.zeros((nbrx, nbrt))

# Fonction solution exacte
def u_exact(x, t):
    if t == 0:
        return np.exp(-x**2 / 0.1)
    return np.exp(-x**2 / (4 * d * t)) / np.sqrt(4 * np.pi * d * t)

# Terme source f(x, t)
def f(x, t):
    if t == 0:
        return 0  
    return np.exp(-x**2 / (4 * t)) * (
        (1 - d) * (x**2 - 2 * t) / (8 * np.sqrt(np.pi) * t**2 * np.sqrt(t))
        - c * x / (4 * np.sqrt(np.pi) * t * np.sqrt(t))
    )

# Conditions initiales
solutionsApprochées[:, 0] = [u_exact(xi, 0) for xi in x]

# Conditions aux limites
for j in range(nbrt):
    solutionsApprochées[0, j] = 0
    solutionsApprochées[-1, j] = 0

# Résolution avec différences finies
for n in range(nbrt - 1):
    for i in range(1, nbrx - 1):
        if c >= 0:
            convection = -c * (solutionsApprochées[i, n] - solutionsApprochées[i - 1, n]) / dx
        else:
            convection = -c * (solutionsApprochées[i + 1, n] - solutionsApprochées[i, n]) / dx

        diffusion = d * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n]) / dx**2
        source = f(x[i], t[n])
        solutionsApprochées[i, n + 1] = solutionsApprochées[i, n] + dt * (diffusion + convection + source)

# Calcul de la solution exacte
for n in range(nbrt):
    for i in range(nbrx):
        solutionExacte[i, n] = u_exact(x[i], t[n])

# Animation des solutions au fil du temps
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

# Tracé de l'écart absolu
plt.figure(figsize=(10, 6))
plt.imshow(ecart_absolu, extent=[0, Nt, -5, 5], origin="lower", aspect="auto", cmap="seismic")
plt.colorbar(label="Écart absolu")
plt.title("Écart absolu entre la solution approchée et la solution exacte")
plt.xlabel("Temps (t)")
plt.ylabel("Espace (x)")
plt.show()

###########################################################################################################################
#On refait les calculs pour obtenir l'erreur en norme infinie en fonction de différents dt


# Liste des différents pas temporels à tester
dt_values = [min(gamma * (dx**2) / d, nu * dx / abs(c)) / 1.2,  
             min(gamma * (dx**2) / d, nu * dx / abs(c)) / 2,   
             min(gamma * (dx**2) / d, nu * dx / abs(c)) / 3,
             min(gamma * (dx**2) / d, nu * dx / abs(c)) / 4]   

# Initialisation des erreurs
errors_Linf = []

# Résolution pour chaque dt
for dt in dt_values:
    nbrt = int(Nt / dt) + 1
    t = np.linspace(0, Nt, nbrt)

    # Initialisation des solutions
    solutionsApprochées = np.zeros((nbrx, nbrt))
    solutionExacte = np.zeros((nbrx, nbrt))

    # Conditions initiales
    solutionsApprochées[:, 0] = [u_exact(xi, 0) for xi in x]

    # Conditions aux limites
    for j in range(nbrt):
        solutionsApprochées[0, j] = 0
        solutionsApprochées[-1, j] = 0

    # Résolution avec différences finies
    for n in range(nbrt - 1):
        for i in range(1, nbrx - 1):
            if c >= 0:
                convection = -c * (solutionsApprochées[i, n] - solutionsApprochées[i - 1, n]) / dx
            else:
                convection = -c * (solutionsApprochées[i + 1, n] - solutionsApprochées[i, n]) / dx

            diffusion = d * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n]) / dx**2
            source = f(x[i], t[n])
            solutionsApprochées[i, n + 1] = solutionsApprochées[i, n] + dt * (diffusion + convection + source)

    # Calcul de la solution exacte
    for n in range(nbrt):
        for i in range(nbrx):
            solutionExacte[i, n] = u_exact(x[i], t[n])

    # Calcul de l'erreur en norme infinie L∞ pour ce dt
    erreur_Linf = np.max(np.abs(solutionsApprochées - solutionExacte))
    errors_Linf.append(erreur_Linf)

# Tracé des erreurs L∞ en fonction de dt (log-log)
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