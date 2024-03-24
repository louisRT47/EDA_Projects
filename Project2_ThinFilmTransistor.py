import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def modelo(Vgs, Vt, k, m):
    return np.piecewise(Vgs, [Vgs < Vt, Vgs >= Vt], [lambda Vgs: 0,
lambda Vgs: k * (W / L) * (Vgs - Vt) ** m])


def get_mVt(Vgs, Vt, m):
 return np.piecewise(Vgs, [Vgs < Vt, Vgs >= Vt], [lambda Vgs: 0,
lambda Vgs: (Vgs - Vt) / m])


def get_k(Vgs, Vt, k, m):
 return np.piecewise(Vgs, [Vgs < Vt, Vgs >= Vt], [lambda Vgs: 0,
lambda Vgs: k * (W / L) * (Vgs - Vt) ** m])


'''Com o Vt determinado'''


def get_m(Vgs, m):
 return np.piecewise(Vgs, [Vgs < Vt, Vgs >= Vt], [lambda Vgs: 0,
lambda Vgs: (Vgs - Vt) / m])


'''********** Leitura dos ficheiros **********'''
'''L=5'''
df_sat_5 = pd.read_csv('P1G4_W50L5_Transfer_saturation.csv', header=0,
sep=",", skiprows=range(0, 251))

Id_sat_5 = np.array(df_sat_5[" ID"])
Vgs_sat_5 = np.array(df_sat_5[" VG"])
Vds_sat_5 = np.array(df_sat_5[" VD"])

df_lin_5 = pd.read_csv('P1G4_W50L5_Transfer_linear.csv', header=0,
sep=",", skiprows=range(0, 251))

Id_lin_5 = np.array(df_lin_5[" ID"])
Vgs_lin_5 = np.array(df_lin_5[" VG"])
Vds_lin_5 = np.array(df_lin_5[" VD"])

'''L=10'''
df_sat_10 = pd.read_csv('P1G4_W50L10_Transfer_saturation.csv',
header=0, sep=",", skiprows=range(0, 251))

Id_sat_10 = np.array(df_sat_10[" ID"])
Vgs_sat_10 = np.array(df_sat_10[" VG"])
Vds_sat_10 = np.array(df_sat_10[" VD"])

df_lin_10 = pd.read_csv('P1G4_W50L10_Transfer_linear.csv', header=0,
sep=",", skiprows=range(0, 251))

Id_lin_10 = np.array(df_lin_10[" ID"])
Vgs_lin_10 = np.array(df_lin_10[" VG"])
Vds_lin_10 = np.array(df_lin_10[" VD"])

'''L=20'''
df_sat_20 = pd.read_csv('P1G4_W50L20_Transfer_saturation.csv',
header=0, sep=",", skiprows=range(0, 251))

Id_sat_20 = np.array(df_sat_20[" ID"])
Vgs_sat_20 = np.array(df_sat_20[" VG"])
Vds_sat_20 = np.array(df_sat_20[" VD"])

df_lin_20 = pd.read_csv('P1G4_W50L20_Transfer_linear.csv', header=0,
sep=",", skiprows=range(0, 251))

Id_lin_20 = np.array(df_lin_20[" ID"])
Vgs_lin_20 = np.array(df_lin_20[" VG"])
Vds_lin_20 = np.array(df_lin_20[" VD"])

'''********** Características de Transferência **********'''

plt.figure(1)
plt.grid()
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Id - Corrente de Dreno [A]')
plt.plot(Vgs_lin_5, Id_lin_5, color='red', label='L = 5um')
plt.plot(Vgs_lin_10, Id_lin_10, color='blue', label='L = 10um')
plt.plot(Vgs_lin_20, Id_lin_20, color='green', label='L = 20um')
plt.legend()
plt.legend(loc='upper left', frameon=True)
plt.title('Gráfico Id em função de Vgs na zona linear')

plt.figure(2)
plt.grid()
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Id - Corrente de Dreno [A]')
plt.plot(Vgs_sat_5, Id_sat_5, color='red', label='L = 5um')
plt.plot(Vgs_sat_10, Id_sat_10, color='blue', label='L = 10um')
plt.plot(Vgs_sat_20, Id_sat_20, color='green', label='L = 20um')
plt.legend()
plt.legend(loc='upper left', frameon=True)
plt.title('Gráfico Id em função de Vgs na zona de saturação')

'''********** Parte 1 **********'''
'''Ex 2'''
# a)
gm5 = np.diff(Id_sat_5) / np.diff(Vgs_sat_5)
gm10 = np.diff(Id_sat_10) / np.diff(Vgs_sat_10)
gm20 = np.diff(Id_sat_20) / np.diff(Vgs_sat_20)

# b)
Idgm5 = Id_sat_5[1:] / gm5
Idgm10 = Id_sat_10[1:] / gm10
Idgm20 = Id_sat_20[1:] / gm20

# d)
Vt_m_5, xy1 = curve_fit(get_mVt, Vgs_sat_5[1:], Idgm5)
Vt1_5 = Vt_m_5[0]
m1_5 = Vt_m_5[1]

Vt_m_10, xy1 = curve_fit(get_mVt, Vgs_sat_10[1:], Idgm10)
Vt1_10 = Vt_m_10[0]
m1_10 = Vt_m_10[1]

Vt_m_20, xy1 = curve_fit(get_mVt, Vgs_sat_20[1:], Idgm20)
Vt1_20 = Vt_m_20[0]
m1_20 = Vt_m_20[1]

# e)
W = 50
L = 5
k5, xy = curve_fit(get_k, Vgs_sat_5, Id_sat_5)
k1_5 = k5[1]
Id1_modelo_5 = modelo(Vgs_sat_5, Vt1_5, k1_5, m1_5)

L = 10
k10, xy = curve_fit(get_k, Vgs_sat_10, Id_sat_10)
k1_10 = k10[1]
Id1_modelo_10 = modelo(Vgs_sat_10, Vt1_10, k1_10, m1_10)

L = 20
k20, xy = curve_fit(get_k, Vgs_sat_20, Id_sat_20)
k1_20 = k20[1]
Id1_modelo_20 = modelo(Vgs_sat_20, Vt1_20, k1_20, m1_20)

# f)
print('')
print('******* PARTE 1 *******')
print('')
print(' L=5um L=10um
L=20')
print('')
print('Vt =', Vt1_5, ' Vt =', Vt1_10, ' Vt =', Vt1_20)
print('m =', m1_5, ' m =', m1_10, ' m =', m1_20)
print('k =', k1_5, ' k =', k1_10, ' k =', k1_20)
print('')

'''********** Parte 2 **********'''
'''Ex2'''
# c)
gm5 = np.diff(Id_lin_5) / np.diff(Vgs_lin_5)
gm10 = np.diff(Id_lin_10) / np.diff(Vgs_lin_10)
gm20 = np.diff(Id_lin_20) / np.diff(Vgs_lin_20)

dgm5 = np.diff(gm5) / np.diff(Vgs_lin_5[1:])
dgm10 = np.diff(gm10) / np.diff(Vgs_lin_10[1:])
dgm20 = np.diff(gm20) / np.diff(Vgs_lin_20[1:])

plt.figure(3)
plt.grid()

plt.xlabel('vgs')
plt.ylabel('dgm')
plt.plot(Vgs_lin_5[2:], dgm5, color='red', label='L=5um')
plt.plot(Vgs_lin_10[2:], dgm10, color='blue', label='L=10um')
plt.plot(Vgs_lin_20[2:], dgm20, color='green', label='L=20um')
plt.legend()
plt.legend(loc='upper right', frameon=True)
plt.title('Gráfico da derivada de gm em função de Vgs (linear)')

# d)
maxIndex5 = np.argmax(dgm5)
maxIndex10 = np.argmax(dgm10)
maxIndex20 = np.argmax(dgm20)

Vt5 = Vgs_lin_5[maxIndex5 + 2]
Vt10 = Vgs_lin_10[maxIndex10 + 2]
Vt20 = Vgs_lin_20[maxIndex20 + 2]

'''Ex3'''
# Valores de Saturação
Vgs_sat_Vt_5 = Vgs_sat_5[maxIndex5:]
Vgs_sat_Vt_10 = Vgs_sat_10[maxIndex10:]
Vgs_sat_Vt_20 = Vgs_sat_20[maxIndex20:]

Id_sat_Vt_5 = Id_sat_5[maxIndex5:]
Id_sat_Vt_10 = Id_sat_10[maxIndex10:]
Id_sat_Vt_20 = Id_sat_20[maxIndex20:]

gm5 = np.diff(Id_sat_Vt_5) / np.diff(Vgs_sat_Vt_5)
gm10 = np.diff(Id_sat_Vt_10) / np.diff(Vgs_sat_Vt_10)
gm20 = np.diff(Id_sat_Vt_20) / np.diff(Vgs_sat_Vt_20)

Idgm5 = Id_sat_Vt_5[1:] / gm5
Idgm10 = Id_sat_Vt_10[1:] / gm10
Idgm20 = Id_sat_Vt_20[1:] / gm20

L = 5
Vt = Vt5
m5, xy = curve_fit(get_m, Vgs_sat_Vt_5[1:], Idgm5)
m2_5 = m5[0]
k5, xy = curve_fit(get_k, Vgs_sat_5, Id_sat_5)
k2_5 = k5[1]

L = 10
Vt = Vt10
m10, xy = curve_fit(get_m, Vgs_sat_Vt_10[1:], Idgm10)
m2_10 = m10[0]
k10, xy = curve_fit(get_k, Vgs_sat_10, Id_sat_10)
k2_10 = k10[1]

L = 20
Vt = Vt20
m20, xy = curve_fit(get_m, Vgs_sat_Vt_20[1:], Idgm20)
m2_20 = m20[0]
k20, xy = curve_fit(get_k, Vgs_sat_20, Id_sat_20)
k2_20 = k20[1]

print('')
print('******* PARTE 2 *******')
print('')

print(' L=5um L=10um
L=20')
print('')
print('Vt =', Vt5, ' Vt =', Vt10, '
Vt =', Vt20)
print('m =', m2_5, ' m =', m2_10, ' m =', m2_20)
print('k =', k2_5, ' k =', k2_10, ' k =', k2_20)
print('')

Id2_modelo_5 = modelo(Vgs_sat_5, Vt5, k2_5, m2_5)
Id2_modelo_10 = modelo(Vgs_sat_10, Vt10, k2_10, m2_10)
Id2_modelo_20 = modelo(Vgs_sat_20, Vt20, k2_20, m2_20)

plt.figure(4)
plt.grid()
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Id - Corrente de Dreno [A]')
plt.plot(Vgs_sat_5, Id1_modelo_5, color='red', label='Método 1')
plt.plot(Vgs_sat_5, Id2_modelo_5, color='blue', label='Método 2')
plt.plot(Vgs_sat_5, Id_sat_5, color='green', label='Valores medidos')
plt.legend()
plt.legend(loc='upper left', frameon=True)
plt.title('Gráfico Id em função de Vgs para L=5um')

plt.figure(5)
plt.grid()
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Id - Corrente de Dreno [A]')
plt.plot(Vgs_sat_10, Id1_modelo_10, color='red', label='Método 1')
plt.plot(Vgs_sat_10, Id2_modelo_10, color='blue', label='Método 2')
plt.plot(Vgs_sat_10, Id_sat_10, color='green', label='Valores
medidos')
plt.legend()
plt.legend(loc='upper left', frameon=True)
plt.title('Gráfico Id em função de Vgs para L=10um')

plt.figure(6)
plt.grid()
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Id - Corrente de Dreno [A]')
plt.plot(Vgs_sat_20, Id1_modelo_20, color='red', label='Método 1')
plt.plot(Vgs_sat_20, Id2_modelo_20, color='blue', label='Método 2')
plt.plot(Vgs_sat_20, Id_sat_20, color='green', label='Valores
medidos')
plt.legend()
plt.legend(loc='upper left', frameon=True)
plt.title('Gráfico Id em função de Vgs para L=20um')

'''********** Erros **********'''

absErr1_5 = np.abs(Id_sat_5 - Id1_modelo_5)
absErr2_5 = np.abs(Id_sat_5 - Id2_modelo_5)
relErr1_5 = np.abs((absErr1_5 / Id_sat_5) * 100)
relErr2_5 = np.abs((absErr2_5 / Id_sat_5) * 100)

absErr1_10 = np.abs(Id_sat_10 - Id1_modelo_10)
absErr2_10 = np.abs(Id_sat_10 - Id2_modelo_10)
relErr1_10 = np.abs((absErr1_10 / Id_sat_10) * 100)
relErr2_10 = np.abs((absErr2_10 / Id_sat_10) * 100)

absErr1_20 = np.abs(Id_sat_20 - Id1_modelo_20)
bsErr2_20 = np.abs(Id_sat_20 - Id2_modelo_20)
relErr1_20 = np.abs((absErr1_20 / Id_sat_20) * 100)
relErr2_20 = np.abs((absErr2_20 / Id_sat_20) * 100)

plt.figure(7)
plt.title('Erros absolutos da modelação') ##falta mencionar o L no
titulo
plt.plot(Vgs_sat_5, absErr1_5, '.', color='purple', label='Método 1 -
L=5um', markersize=3)
plt.plot(Vgs_sat_5, absErr2_5, '.', color='orange', label='Método 2 -
L=5um', markersize=3)
plt.plot(Vgs_sat_10, absErr1_10, '.', color='blue', label='Método 1 -
L=10um', markersize=3)
plt.plot(Vgs_sat_10, absErr2_10, '.', color='green', label='Método 2 -
L=10um', markersize=3)
plt.plot(Vgs_sat_20, absErr1_20, '.', color='black', label='Método 1 -
L=20um', markersize=3)
plt.plot(Vgs_sat_20, absErr2_20, '.', color='red', label='Método 2 -
L=20um', markersize=3)
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Erro Absoluto [A]')
plt.grid()
plt.legend()

plt.figure(8)
plt.title('Erros relativos da modelação para L=5um')
plt.plot(Vgs_sat_5, relErr1_5, '.', color='blue', label='Método 1 -
L=5um', markersize=3)
plt.plot(Vgs_sat_5, relErr2_5, '.', color='red', label='Método 2 -
L=5um', markersize=3)
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Erro Relativo [%]')
plt.grid()
plt.legend()

plt.figure(9)
plt.title('Erros relativos da modelação para L=10um')
plt.plot(Vgs_sat_10, relErr1_10, '.', color='blue', label='Método 1 -
L=10um', markersize=3)
plt.plot(Vgs_sat_10, relErr2_10, '.', color='red', label='Método 2 -
L=10um', markersize=3)
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Erro Relativo [%]')
plt.grid()
plt.legend()

plt.figure(10)
plt.title('Erros relativos da modelação para L=20um')
plt.plot(Vgs_sat_20, relErr1_20, '.', color='blue', label='Método 1 -
L=20um', markersize=3)
plt.plot(Vgs_sat_20, relErr2_20, '.', color='red', label='Método 2 -
L=20um', markersize=3)
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Erro Relativo [%]')
plt.grid()
plt.legend()

'''********** Parte 3 **********'''
'''Ex 1'''
def tft(data, k, Rs, Vt, m):
    Vg, Id = data
    v_limiar = Rs * Id + Vt
    j = np.where(Vg > v_limiar)[0][0]
    return np.piecewise(Vg, [Vg >= v_limiar, Vg < v_limiar],
    [lambda Vg: k * (Vg - Id[j:] * Rs - Vt) ** m,
    lambda Vg: 0])

df_sat = pd.read_csv('P1G4_W50L20_Transfer_saturation.csv', header=0,
sep=",", skiprows=range(0, 251))
Id_sat = np.array(df_sat[" ID"])
Vg_sat = np.array(df_sat[" VG"])

parameters, covariance = curve_fit(tft, (Vg_sat, Id_sat), Id_sat,
maxfev=4000,
bounds=([0, 20, -0.5, 2], [1e-3,
1e3, 0.5, 3]), p0=(1e-6, 200, 0.1, 2.2))

K = parameters[0]
Rs = parameters[1]
Vt = parameters[2]
m = parameters[3]

print('K =', K)
print('Rs =', Rs)
print('Vt =', Vt)
print('m =', m)

'''Ex 2'''
from scipy.optimize import fsolve

def get_implicit(Id, Vg):
if Vg - Vt - Rs * Id < 0:
return 0
else:
return (Id - K * (Vg - Vt - Rs * Id) ** m)

f = lambda Id: get_implicit(Id, Vg)

Id_s = np.array([])
for Vg in Vg_sat:
idx = fsolve(f, 0)
Id_s = np.append(Id_s, idx)

plt.figure(11)
plt.grid()
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Id - Corrente Dreno [A]')
plt.plot(Vg_sat, Id_s, color='red', label='Característica obtida')
plt.plot(Vg_sat, Id_sat, color='green', label='Valores medidos')
plt.legend()
plt.legend(loc='upper left', frameon=True)
plt.title('Gráfico Id em função de Vgs para L=20um')

'''********** Erros **********'''
absolute_error = np.abs(Id_sat - Id_s)
relative_error = np.abs((absolute_error / Id_sat) * 100)

plt.figure(12)
plt.grid()
plt.title('Erro absoluto da modelação para L=20um')
plt.plot(Vg_sat, absolute_error, '.', color='blue', markersize=3)
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Erro Absoluto [A]')

plt.figure(13)
plt.title('Erro relativo da modelação para L=20um')
plt.plot(Vg_sat, relative_error, '.', color='red', markersize=3)
plt.xlabel('Vgs - Tensão Gate-Source [V]')
plt.ylabel('Erro Relativo [%]')
plt.grid()