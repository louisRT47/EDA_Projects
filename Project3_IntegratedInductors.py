# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:28:52 2021

@author: Luisr
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize 

#TechfileUMC130nm 
miuR=1; 
miu=4*math.pi*1e-7*miuR; 
E0= 8.854187817e-12;                #F/m
c=299792458;                        #Layer Oxide(Metal)   #m/s
t= 2.8e-6;                          #Metalthickness(m)
Rsheet=10*1e-3;                     #SheetResistance[Ohm/square](=Ro/t)
Ro=Rsheet*t;                        #MetalResistance[Ohm.m]
Toxide = 5.42e-6;                   #Dielectric(Oxide)Thickness(frommetaltosubstrate)
t_M_underpass = 0.4e-6; 
toxide_Munderpass = Toxide-t_M_underpass-4.76e-6;
Erox= 4;                            #OxideEr
Eox=E0*Erox;                        #Oxidepermitivity%Substrate
Ersub=11.9;                         #substrateEr
Esub=E0*Ersub;                      #substratepermititvity
Tsub =700e-6;                       #substratethickness
Sub_resistiv=2800;                  #28ohm-cmou2800Ohm-m


Nside = [4, 6, 8]
dout = 340e-6
L_tecnologia =130e-9  #Tecnologia 130nm
W_tecnologia = 10e-6
fr =1e9
nTurns =3
s =5e-6
K1 = [2.34, 2.33, 2.25]
K2 = [2.75, 3.82, 3.55]
Kshape = ["Square", "Hexagonal", "Octagonal"]


''' Parte 1 '''

#Calcular o valor dos componentes em função dos W e L tecnologicos

def get_Rsub(W, dout, Nside, nTurns, s):      # função para obtenção Rsub
    din = dout - (2*nTurns*W) - 2*(nTurns -1)*s
    davg = 0.5*(dout +din)
    l = Nside * davg * nTurns * (math.tan(math.pi/Nside))
    return 2*Tsub *(Sub_resistiv)/(W*l)
                                      
def get_Rs(Nside, nTurns, W, dout, s ,freq): # função para obtenção do Rs
    din = dout - (2*nTurns*W) - 2*(nTurns -1)*s
    davg = 0.5*(dout +din)
    l = Nside * davg * nTurns * (math.tan(math.pi/Nside))
    sigma = math.sqrt(Ro/(np.pi*freq*miu))
    t_eff = (1-math.exp(-t/sigma))
    return (l*Ro)/(W*sigma*t_eff)

def get_Cox(E_x, t_x, dout, nTurns, Nside,s): # função para obtenção do Cox e do Csub
    din = dout - (2*nTurns*W_tecnologia) - 2*(nTurns -1)*s
    davg = 0.5*(dout +din)
    l = Nside * davg * nTurns * (math.tan(math.pi/Nside))
    return 0.5*l*W_tecnologia *(E_x/t_x)

def get_Cp(nTurns, W):      # função para obtenção do Cp
    return (nTurns-1) * (W**2) * (Eox/toxide_Munderpass)

def get_Ls(W, K1, K2, nTurns, dout, s): # função para obtenção do Ls
    din = dout - (2*nTurns*W) - 2*(nTurns -1)*s
    davg = 0.5*(dout+din)
    px = (dout-din)/(dout+din)
    return ((nTurns**2)*davg * K1*miu)/(1+K2*px)
    


''' Parte 2'''

def get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, freq):
    w = 2*np.pi*freq 
    Z1 = ((1/(Rs+ (1j*w*Ls))) + 1j*w*Cp)**(-1)
    Z2 = (((1/Rsub) + 1j*w*Csub)**(-1)) + (1/(1j*w*Cox))
    return (Z1*Z2)/(Z1+Z2)
    

'''1'''
Ztotal = np.zeros(1000)
inductance = np.zeros(1000)
for i in range(0, 3):    
    Cp = get_Cp(nTurns, W_tecnologia)
    Cox = get_Cox(Eox, Toxide , dout, nTurns, Nside[i], s)
    Csub = get_Cox(Esub, Tsub, dout, nTurns, Nside[i], s)
    Rs = get_Rs(Nside[i], nTurns, W_tecnologia, dout, s, fr)
    Rsub = get_Rsub(W_tecnologia, dout, Nside[i], nTurns, s)

    Ls = get_Ls(W_tecnologia, K1[i], K2[i], nTurns, dout, s)

    # Calculo do Ztotal com o no 1 ligado a massa
    Ztotal = get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, fr)
    inductance = np.imag(Ztotal)/(2*math.pi*fr)

    print("Geometry -->", Kshape[i] ,"\n")
    print('Rsub: ', Rsub)
    print('Rs: ', Rs)
    print('Cox: ', Cox)
    print('Csub: ', Csub)
    print('Cp: ', Cp)
    print('Ls: ', Ls)
    print('\n')
    print("Ztotal: "+ str(Ztotal))
    print("Inductance at 1Ghz: " + str(np.imag(Ztotal)/(2*math.pi*fr)), "\n")
        

'''2'''

fr =2.4e9
for i in range(0, 3):
    
    Cp = get_Cp(nTurns, W_tecnologia)
    Cox = get_Cox(Eox, Toxide , dout, nTurns, Nside[i], s)
    Csub = get_Cox(Esub, Tsub, dout, nTurns, Nside[i], s)
    Rs = get_Rs(Nside[i], nTurns, W_tecnologia, dout, s, fr)
    Rsub = get_Rsub(W_tecnologia, dout, Nside[i], nTurns, s)
    Ls = get_Ls(W_tecnologia, K1[i], K2[i], nTurns, dout, s)

    # Calculo do Ztotal com o no 1 ligado a massa
    Ztotal = get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, fr)

    print("------------------ Geometry = ", Kshape[i], "------------------\n")
    QF2_4GHZ_only = np.imag(Ztotal)/np.real(Ztotal)
    print("Fator de qualidade a 2,4GHz: %.4f" % (QF2_4GHZ_only), "\n")


print('Rs a 2.4GHz: ', Rs)
print('\n')


'''3, 4 e 5'''
Ztotal = np.zeros(1000)
inductance = np.zeros((3,1000))
freqmax= np.logspace(4,11,int(1e3))
QF2_4GHZ = np.zeros((3,1000))

for i in range(0,3):
    Cp = get_Cp(nTurns, W_tecnologia)
    Cox = get_Cox(Eox, Toxide , dout, nTurns, Nside[i], s)
    Csub = get_Cox(Esub, Tsub, dout, nTurns, Nside[i], s)
    Rsub = get_Rsub(W_tecnologia, dout, Nside[i], nTurns, s)
    Ls = get_Ls(W_tecnologia, K1[i], K2[i], nTurns, dout, s)
    k = 0    
    for f in freqmax:
        #Square geometry
        Rs = get_Rs(Nside[i], nTurns, W_tecnologia, dout, s, f)
    
        # Calculo do Ztotal com o no 1 ligado a massa
        Ztotal = get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, f)
        inductance[i,k] = np.imag(Ztotal)/(2*math.pi*f)
        QF2_4GHZ[i,k] = np.imag(Ztotal)/np.real(Ztotal)
        k=k+1
        

plt.figure(1)
plt.title("Gráfico da Indutância para\n as Várias Geometrias")
plt.semilogx(freqmax, inductance[0,], color = 'red', label = 'Quadrada')
negInd = np.where(inductance[0,] < 0)[0][0]
ressFreq = 0.5* (freqmax[negInd] + freqmax[negInd-1])
print('------------------ Geometry =  Square ------------------\n')
print('Frequênçia de ressonânçia: %e' % (ressFreq))
plt.semilogx(ressFreq,0, 'x', color = 'black', markersize = 12, label = 'Freq. Ressonânçia')
plt.semilogx(freqmax, inductance[1,], color = 'blue', label = 'Hexagonal')
negInd = np.where(inductance[1,] < 0)[0][0]
ressFreq = 0.5* (freqmax[negInd] + freqmax[negInd-1])
print('------------------ Geometry =  Hexagonal ------------------\n')
print('Frequênçia de ressonânçia: %e' % (ressFreq))
plt.semilogx(ressFreq,0, 'x', color = 'black', markersize = 12)
plt.semilogx(freqmax, inductance[2,], color = 'green', label = 'Octogonal')
negInd = np.where(inductance[2,] < 0)[0][0]
ressFreq = 0.5* (freqmax[negInd] + freqmax[negInd-1])
print('------------------ Geometry =  Octagonal ------------------\n')
print('Frequênçia de ressonânçia: %e' % (ressFreq))
plt.semilogx(ressFreq,0, 'x', color = 'black', markersize = 12)
plt.xlabel('frequência [Hz]')
plt.ylabel('Indutância')
plt.grid()
plt.legend()


plt.figure(2)
plt.title("Gráfico do Fator de Qualidade para as Várias Geometrias")
plt.semilogx(freqmax, QF2_4GHZ[0,], color = 'red', label = 'Quadrada')
plt.semilogx(freqmax, QF2_4GHZ[1,], color = 'blue', label = 'Hexagonal')
plt.semilogx(freqmax, QF2_4GHZ[2,], color = 'green', label = 'Octogonal')
plt.semilogx(2.4e9, QF2_4GHZ_only, '+', color = 'black', markersize = 15, label = 'f = 2,4GHz')
plt.xlabel('frequência [Hz]')
plt.ylabel('Fator de Qualidade')
plt.grid()
plt.legend()
plt.axis([10**4,10**11,-50,20])



''' Parte 3 '''
# LTspice


''' Parte 4 '''


# Função de obtenção do fator de qualidade
def get_qualityFactor(Nside, fr, i, nTurns, dout, w):
    Cp = get_Cp(nTurns, w)
    Cox = get_Cox(Eox, Toxide , dout, nTurns, Nside, s)
    Csub = get_Cox(Esub, Tsub, dout, nTurns, Nside, s)
    Rs = get_Rs(Nside, nTurns, w, dout, s, fr)
    Rsub = get_Rsub(w, dout, Nside, nTurns, s)
    Ls = get_Ls(w, K1[i], K2[i], nTurns, dout, s)
    Ztotal = get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, fr)
    return np.imag(Ztotal)/np.real(Ztotal)



# Função de obtenção do valor da indutancia para uma frequênçia variavel
def get_Inductance_a(Nside, fr, i, nTurns, dout, w):
    Cp = get_Cp(nTurns, w)
    Cox = get_Cox(Eox, Toxide , dout, nTurns, Nside, s)
    Csub = get_Cox(Esub, Tsub, dout, nTurns, Nside, s)
    Rsub = get_Rsub(w, dout, Nside, nTurns, s)
    Ls = get_Ls(w, K1[i], K2[i], nTurns, dout, s)
    inductance =np.array([])
    for f in fr:
        Rs =  get_Rs(Nside, nTurns, w, dout, s, f)
        Ztotal = get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, f)
        inductance = np.append(inductance, np.imag(Ztotal)/(2*math.pi*f))
    return inductance



# Função de obtenção do valor da indutancia para um frequênçia fixa
def get_Inductance(Nside, fr, i, nTurns, dout, w):
    Cp = get_Cp(nTurns, w)
    Cox = get_Cox(Eox, Toxide , dout, nTurns, Nside, s)
    Csub = get_Cox(Esub, Tsub, dout, nTurns, Nside, s)
    Rs = get_Rs(Nside, nTurns, w, dout, s, fr)
    Rsub = get_Rsub(w, dout, Nside, nTurns, s)
    Ls = get_Ls(w, K1[i], K2[i], nTurns, dout, s)
    Ztotal = get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, fr)
    inductanceZ = np.imag(Ztotal)/(2*math.pi*fr)
    return inductanceZ



#Função de optimização da impedânçia
def get_L_optimize(x):
    
    Nside = 8 #octogonal
    i = 2 # posição 2 que equivale a Nside = 8
    
    freq = 2.4e9
    inductance_op = 4.5e-9   #optimização para um L =4,5nH
    dout, n, w  = x[0], x[1], x[2]
   
    result = get_Inductance(Nside, freq, i, n, dout, w)
    return abs(result - inductance_op)



# Função dos bounds da frequênçia de ressonânçia
def opti_ressFreq(x):
    
    Nside = 8
    i = 2 # posição 2 que que equivale a Nside = 8
    dout, n, w = x[0], x[1], x[2]
    
    #criar um vetor de frequênçias
    fs_vect = np.logspace(4, 11, int(1e3))
    
    Z_vect = np.array([])
    Z_vect = get_Inductance_a(Nside, fs_vect, i, n, dout, w)
        
        
    #negIndex = np.where(Z_vect < 0)[0] [0]
    #fRess = 0.5* (fs_vect[negIndex] + fs_vect[negIndex-1])
    # Verificar onde a fase é zero e fazer a média, tipo de um aproximação da fRessonância
    for k in range(len(Z_vect)):
        if(Z_vect[k]*Z_vect[k-1]<0):
            fRess=0.5* (fs_vect[k] + fs_vect[k-1])
    
    
    #min_ressFreq = (2.4e9) #alina a
    min_ressFreq = (2.4e9) * 10 #alinea c
    optim = fRess - min_ressFreq
    return optim



# Parametros e minimização da função get_L_optimize
a0 = [340e-6, 3, 10e-6]  # pos0 = Dout, pos1 =n, pos2 = W

bnds = ((200e-6, 600e-6), (1, 5), (5e-6, 50e-6))

opts = {'disp': 0, 'maxiter': 2000}
cons = ({'type': 'ineq', 'fun': opti_ressFreq})

result = minimize(get_L_optimize, x0 = a0, options = opts, tol = 1e-15, bounds = bnds, constraints = cons)



# Parametros do minimize
solution = result.x
evaluation = result.fun
dout = solution[0]
n = solution[1]
w =  solution[2]
i = 2
f = 2.4e9

print("\nNumber of evaluations: %d" % result["nfev"])
print("dout = %e, n = %.2f, w = %e" % (dout, n, w))
print("Evaluation = %e" % evaluation)

Nside_op = Nside[i]
inductance = get_Inductance(Nside_op,f,i,n,dout,w)
print("Inductance at 2,4Ghz:  %e H" %(inductance))

QF2_4GHZ_opti = get_qualityFactor(Nside_op, f, i, n, dout, w)
print("Quality factor at 2,4GHz: %.4f" % (QF2_4GHZ_opti))





vectInduct = np.zeros(1000)
Nside = 8
vectInduct = get_Inductance_a(Nside, freqmax, 2, n, dout, w)

plt.figure(3)
plt.title("Gráfico da Indutância após a optimização")
plt.semilogx(freqmax, vectInduct, color = 'purple', label = 'Octogonal')
negInd = np.where(vectInduct < 0)[0][0]
ressFreq = 0.5* (freqmax[negInd] + freqmax[negInd-1])
plt.semilogx(ressFreq,0, 'x', color = 'black', markersize = 12, label = 'Freq. Ressonânçia')
plt.xlabel('frequência [Hz]')
plt.ylabel('Indutância')
plt.grid()
plt.legend()

print("Frequencia de ressonancia: %e" % (ressFreq))



QF2_4GHZ_multi = np.zeros(1000)
def get_Ztotal_opti(Nside, fr, i, nTurns, dout, w):
    Cp = get_Cp(nTurns, w)
    Cox = get_Cox(Eox, Toxide , dout, nTurns, Nside, s)
    Csub = get_Cox(Esub, Tsub, dout, nTurns, Nside, s)
    Rsub = get_Rsub(w, dout, Nside, nTurns, s)
    Ls = get_Ls(w, K1[i], K2[i], nTurns, dout, s)
    Ztotal =np.array([])
    for f in fr:
        Rs =  get_Rs(Nside, nTurns, w, dout, s, f)
        Ztotal = np.append(Ztotal, get_Ztotal(Cp, Ls, Rs, Cox, Csub, Rsub, f))
    return Ztotal

vectZtotal = get_Ztotal_opti(Nside, freqmax, 2, n, dout, w)
for h in range(0, 1000):
    QF2_4GHZ_multi[h] = np.imag(vectZtotal[h])/np.real(vectZtotal[h])

plt.figure(4)
plt.title("Gráfico do Fator de Qualidade após a optimização")
plt.semilogx(freqmax, QF2_4GHZ_multi, color = 'green', label = 'Octogonal')
plt.semilogx(2.4e9, QF2_4GHZ_opti, '+', color = 'black', markersize = 15, label = 'f = 2.4GHz')
plt.xlabel('frequência [Hz]')
plt.ylabel('Fator de Qualidade')
plt.xlabel('frequência [Hz]')
plt.ylabel('Fator de Qualidade')
plt.grid()
plt.legend()
plt.axis([10**4,10**11,-50,25])
