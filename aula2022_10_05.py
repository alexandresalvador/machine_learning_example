import numpy as np
import matplotlib.pyplot as plt

vetor_teste = [5, 10, 6, 9, -5, 3.8, 8]

#calcula a media
print(np.mean(vetor_teste)) 
print(np.average(vetor_teste))

#calcula a mediana
print(np.median(vetor_teste))

#remove valores repetidos e ordena
print(np.unique(vetor_teste))

#desvio padrao - quão homogeneo é 
print(np.std(vetor_teste))

#maior e menor valor dentro de nosso vetor
print(np.amax(vetor_teste))
print(np.amin(vetor_teste))

#cria array de 9 elementos
#print(np.arange(9))

# como transformar em array - np.array([])
array_a = np.array([2,6,9])
array_b = np.array([8,5,20])

print(np.concatenate((array_a, array_b)))

a = np.array([
    [ 1,  2,  3,  4], [ 5,  6,  7,  8 ], [ 9, 10, 11, 12],
])

print(a[a >16])

#matplotlib

plt.style.use('_mpl-gallery-nogrid')

x = [1, 2, 3, 4]
colors = plt.get_cmap('Reds')(np.linspace(0.2, 0.7, len(x)
))
fig, ax = plt.subplots()

ax.pie(x, colors=colors, radius=6, center=(8,8), frame=True)

ax.set(xlim=(0, 16), xticks=np.arange(1, 16),
       ylim=(0, 16), yticks=np.arange(1, 16))

plt.show()
