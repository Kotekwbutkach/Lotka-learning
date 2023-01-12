import numpy as np
from scipy import integrate


def pochodna_ofiar(X0, t, a, b, c, d):
    poch_x = [(a - b*X0[1])*X0[0]]
    return poch_x


def pochodna_drapieznikow(X0, t, a, b, c, d):
    poch_y = [(c*X0[0] - d)*X0[1]]
    return poch_y


def rownanie_LV(X0, t, a, b, c, d):
    x_pochodne = pochodna_ofiar(X0, t, a, b, c, d)
    y_pochodne = pochodna_drapieznikow(X0, t, a, b, c, d)
    rhs = np.array([x_pochodne, y_pochodne])
    return rhs.ravel()


Nt = 1000  # liczba punktów, w których liczymy liczebność populacji
tmax = 60  # do jakiego czasu liczymy
t = np.linspace(0, tmax, Nt)

rng = np.random.default_rng()

k = 500

dane_treningowe = list()

for i in range(k):
    param, x0 = rng.uniform(1, 10, 4), rng.uniform(1, 10, 2)  # parametry równania
    zwierzeta = integrate.solve_ivp(lambda t, x0: rownanie_LV(x0, t, *param),
                                    (0, 60),
                                    x0,
                                    method='RK45',
                                    t_eval=t,
                                    dense_output=True)
    dane_treningowe.append(zwierzeta.y.ravel("F"))
    if (i + 1) * 50 % k == 0:
        print(f"{i + 1}/{k}")

print(dane_treningowe)
print(dane_treningowe[0].shape)

np.savetxt("dane_treningowe.csv", dane_treningowe, delimiter=";")
