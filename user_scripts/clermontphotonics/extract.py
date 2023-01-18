import numpy as np
import matplotlib.pyplot as plt

n_couches = "20"
runner = "A"
algo = "DE"
function = "BraggO"
budget = 10000

file_name = f"Res1/out_{function}_{algo}_{n_couches}_{budget}_{runner}.npy"
results = np.load(file_name,allow_pickle = True)

#Graphique pour comparaisons
values = []
fig1 = plt.figure(1)
for k in range(len(results)):
    values.append(results[k][1][-1])
sorted = np.sort(values)
plt.plot(sorted)
plt.title(f'Function {function} with {algo}')
plt.show(fig1)
# Représenter les courbes de convergence

plt.clf()
fig2 = plt.figure(2)
for k in range(len(results)):
    plt.plot(results[k][1])
plt.title("Convergences")
plt.show(fig2)
