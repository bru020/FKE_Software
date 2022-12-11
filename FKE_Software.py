import tkinter as tk
from tkinter import ttk
import numpy as np
from numpy.core.numeric import identity
import pandas as pd
from scipy import linalg
import scipy
import matplotlib.pyplot as plt

'''
Códigos das cores utilizadas:
Roxo - #1E0127
Dourado - #DAA520
Amarelo claro - #FFDE93
Lilás - #DDA0DD
Azure - #E0FFFF

Fontes:
Títulos - Times New Roman
Textos - Arial

Ver depois:
1. Botões de voltar nas abas seguintes e conflito com imagem
2. Adicionar variável aos inputs de texto
3. Definir função para os botões dos gráficos
'''

Q = identity(6)
P_est = identity(6)
Rk = identity(4)
x_est = []
N = []
T = []

def abrir_FKE_cond_iniciais():
    menu.withdraw()
    FKE_cond_iniciais = tk.Toplevel()
    FKE_cond_iniciais.title("FKE - Condições iniciais")
    FKE_cond_iniciais.config(background='#1E0127')
    FKE_cond_iniciais.focus_force()
    FKE_cond_iniciais.grab_set()

    frame1 = tk.Frame(FKE_cond_iniciais, bg="#1E0127")
    frame1.pack()

    topico1_cond_iniciais = tk.Label(frame1, text="Condições iniciais", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico1_cond_iniciais.grid(column=0, row=0, padx=10, pady=10)

    texto1_cond_iniciais = tk.Label(frame1, text="Para iniciar os cálculos do FKE, é necessário informar as condições \n"
                                           "iniciais de alguns elementos usados nas equações, como os parâmetros \n"
                                           "das matrizes de covariância Q, P e R, além do vetor de estado inicial.", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto1_cond_iniciais.grid(column=0, row=1, padx=10, pady=10)

    texto2_cond_iniciais = tk.Label(frame1, text="Assim, insira abaixo os seis elementos das diagonais de cada \n"
                                                            "matriz, além dos seis elementos do vetor linha de estado inicial:", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto2_cond_iniciais.grid(column=0, row=2, padx=10, pady=10)

    #Criar a borda
    frame_fundo = tk.Frame(FKE_cond_iniciais, bg='#1E0127')
    frame_fundo.pack()
    frame_borda = tk.Frame(frame_fundo, bg='#FFDE93')
    frame_borda.pack(padx=5, pady=5)

    frame2 = tk.Frame(frame_borda, bg="#1E0127")
    frame2.pack(padx=1, pady=1)

    # Elementos do vetor de estado inicial
    roll_0 = tk.Label(frame2, text="phi =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    roll_0.grid(column=0, row=0, padx=5, pady=10)
    input_roll_0 = tk.Text(frame2, height=1, width=5)
    input_roll_0.grid(column=1, row=0, padx=5, pady=10)
    pitch_0 = tk.Label(frame2, text="roll =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    pitch_0.grid(column=2, row=0, padx=5, pady=10)
    input_pitch_0 = tk.Text(frame2, height=1, width=5)
    input_pitch_0.grid(column=3, row=0, padx=5, pady=10)
    yaw_0 = tk.Label(frame2, text="yaw =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    yaw_0.grid(column=4, row=0, padx=5, pady=10)
    input_yaw_0 = tk.Text(frame2, height=1, width=5)
    input_yaw_0.grid(column=5, row=0, padx=5, pady=10)
    bias_x0 = tk.Label(frame2, text="bias x =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    bias_x0.grid(column=6, row=0, padx=5, pady=10)
    input_bias_x0 = tk.Text(frame2, height=1, width=5)
    input_bias_x0.grid(column=7, row=0, padx=5, pady=10)
    bias_y0 = tk.Label(frame2, text="bias y =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    bias_y0.grid(column=8, row=0, padx=5, pady=10)
    input_bias_y0 = tk.Text(frame2, height=1, width=5)
    input_bias_y0.grid(column=9, row=0, padx=5, pady=10)
    bias_z0 = tk.Label(frame2, text="bias z =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    bias_z0.grid(column=10, row=0, padx=5, pady=10)
    input_bias_z0 = tk.Text(frame2, height=1, width=5)
    input_bias_z0.grid(column=11, row=0, padx=5, pady=10)

    #Elementos da matriz Q
    q11 = tk.Label(frame2, text="q11 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    q11.grid(column=0, row=1, padx=5, pady=10)
    input_q11 = tk.Text(frame2, height=1, width=5)
    input_q11.grid(column=1, row=1, padx=5, pady=10)
    q22 = tk.Label(frame2, text="q22 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    q22.grid(column=2, row=1, padx=5, pady=10)
    input_q22 = tk.Text(frame2, height=1, width=5)
    input_q22.grid(column=3, row=1, padx=5, pady=10)
    q33 = tk.Label(frame2, text="q33 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    q33.grid(column=4, row=1, padx=5, pady=10)
    input_q33 = tk.Text(frame2, height=1, width=5)
    input_q33.grid(column=5, row=1, padx=5, pady=10)
    q44 = tk.Label(frame2, text="q44 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    q44.grid(column=6, row=1, padx=5, pady=10)
    input_q44 = tk.Text(frame2, height=1, width=5)
    input_q44.grid(column=7, row=1, padx=5, pady=10)
    q55 = tk.Label(frame2, text="q55 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    q55.grid(column=8, row=1, padx=5, pady=10)
    input_q55 = tk.Text(frame2, height=1, width=5)
    input_q55.grid(column=9, row=1, padx=5, pady=10)
    q66 = tk.Label(frame2, text="q66 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    q66.grid(column=10, row=1, padx=5, pady=10)
    input_q66 = tk.Text(frame2, height=1, width=5)
    input_q66.grid(column=11, row=1, padx=5, pady=10)

    # Elementos da matriz P
    p11 = tk.Label(frame2, text="p11 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    p11.grid(column=0, row=2, padx=5, pady=10)
    input_p11 = tk.Text(frame2, height=1, width=5)
    input_p11.grid(column=1, row=2, padx=5, pady=10)
    p22 = tk.Label(frame2, text="p22 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    p22.grid(column=2, row=2, padx=5, pady=10)
    input_p22 = tk.Text(frame2, height=1, width=5)
    input_p22.grid(column=3, row=2, padx=5, pady=10)
    p33 = tk.Label(frame2, text="p33 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    p33.grid(column=4, row=2, padx=5, pady=10)
    input_p33 = tk.Text(frame2, height=1, width=5)
    input_p33.grid(column=5, row=2, padx=5, pady=10)
    p44 = tk.Label(frame2, text="p44 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    p44.grid(column=6, row=2, padx=5, pady=10)
    input_p44 = tk.Text(frame2, height=1, width=5)
    input_p44.grid(column=7, row=2, padx=5, pady=10)
    p55 = tk.Label(frame2, text="p55 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    p55.grid(column=8, row=2, padx=5, pady=10)
    input_p55 = tk.Text(frame2, height=1, width=5)
    input_p55.grid(column=9, row=2, padx=5, pady=10)
    p66 = tk.Label(frame2, text="p66 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    p66.grid(column=10, row=2, padx=5, pady=10)
    input_p66 = tk.Text(frame2, height=1, width=5)
    input_p66.grid(column=11, row=2, padx=5, pady=10)

    # Elementos da matriz R
    r11 = tk.Label(frame2, text="r11 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    r11.grid(column=0, row=3, padx=5, pady=10)
    input_r11 = tk.Text(frame2, height=1, width=5)
    input_r11.grid(column=1, row=3, padx=5, pady=10)
    r22 = tk.Label(frame2, text="r22 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    r22.grid(column=2, row=3, padx=5, pady=10)
    input_r22 = tk.Text(frame2, height=1, width=5)
    input_r22.grid(column=3, row=3, padx=5, pady=10)
    r33 = tk.Label(frame2, text="r33 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    r33.grid(column=4, row=3, padx=5, pady=10)
    input_r33 = tk.Text(frame2, height=1, width=5)
    input_r33.grid(column=5, row=3, padx=5, pady=10)
    r44 = tk.Label(frame2, text="r44 =", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    r44.grid(column=6, row=3, padx=5, pady=10)
    input_r44 = tk.Text(frame2, height=1, width=5)
    input_r44.grid(column=7, row=3, padx=5, pady=10)

    def guardar_valor_inicial():
        v_q11 = float(input_q11.get("1.0", "end-1c"))
        v_q22 = float(input_q22.get("1.0", "end-1c"))
        v_q33 = float(input_q33.get("1.0", "end-1c"))
        v_q44 = float(input_q44.get("1.0", "end-1c"))
        v_q55 = float(input_q55.get("1.0", "end-1c"))
        v_q66 = float(input_q66.get("1.0", "end-1c"))
        v_p11 = float(input_p11.get("1.0", "end-1c"))
        v_p22 = float(input_p22.get("1.0", "end-1c"))
        v_p33 = float(input_p33.get("1.0", "end-1c"))
        v_p44 = float(input_p44.get("1.0", "end-1c"))
        v_p55 = float(input_p55.get("1.0", "end-1c"))
        v_p66 = float(input_p66.get("1.0", "end-1c"))
        v_r11 = float(input_r11.get("1.0", "end-1c"))
        v_r22 = float(input_r22.get("1.0", "end-1c"))
        v_r33 = float(input_r33.get("1.0", "end-1c"))
        v_r44 = float(input_r44.get("1.0", "end-1c"))
        v_x11 = float(input_roll_0.get("1.0", "end-1c"))
        v_x22 = float(input_pitch_0.get("1.0", "end-1c"))
        v_x33 = float(input_yaw_0.get("1.0", "end-1c"))
        v_x44 = float(input_bias_x0.get("1.0", "end-1c"))
        v_x55 = float(input_bias_y0.get("1.0", "end-1c"))
        v_x66 = float(input_bias_z0.get("1.0", "end-1c"))
        Q[0, 0] = v_q11
        Q[1, 1] = v_q22
        Q[2, 2] = v_q33
        Q[3, 3] = v_q44
        Q[4, 4] = v_q55
        Q[5, 5] = v_q66
        P_est[0, 0] = v_p11
        P_est[1, 1] = v_p22
        P_est[2, 2] = v_p33
        P_est[3, 3] = v_p44
        P_est[4, 4] = v_p55
        P_est[5, 5] = v_p66
        Rk[0, 0] = v_r11
        Rk[1, 1] = v_r22
        Rk[2, 2] = v_r33
        Rk[3, 3] = v_r44
        x_est.append(v_x11)
        x_est.append(v_x22)
        x_est.append(v_x33)
        x_est.append(v_x44)
        x_est.append(v_x55)
        x_est.append(v_x66)

    frame3 = tk.Frame(FKE_cond_iniciais, bg="#1E0127")
    frame3.pack()

    botao_voltar_FKEpMenu = tk.Button(frame3, text="Voltar para menu", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=lambda: [FKE_cond_iniciais.destroy(), menu.deiconify()])
    botao_voltar_FKEpMenu.grid(column=0, row=0, padx=5, pady=10)

    botao_proximo_cond_iniciais = tk.Button(frame3, text="Próximo", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=lambda: [guardar_valor_inicial(), abrir_FKE_observacoes()])
    botao_proximo_cond_iniciais.grid(column=1, row=0, padx=5, pady=10)

def abrir_FKE_observacoes():
    FKE_obs = tk.Toplevel()
    FKE_obs.title("FKE - Número de observações")
    FKE_obs.config(background='#1E0127')
    FKE_obs.focus_force()
    FKE_obs.grab_set()

    frame1 = tk.Frame(FKE_obs, bg='#1E0127')
    frame1.pack()

    topico1_obs = tk.Label(frame1, text="Número de observações do FKE", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico1_obs.pack(padx=10, pady=10)

    # Criar a borda
    frame_fundo = tk.Frame(FKE_obs, bg='#1E0127')
    frame_fundo.pack()
    frame_borda = tk.Frame(frame_fundo, bg='#FFDE93')
    frame_borda.pack(padx=5, pady=5)

    frame2 = tk.Frame(frame_borda, bg='#1E0127')
    frame2.pack(padx=1, pady=1)

    texto1_obs = tk.Label(frame2, text="Digite o número de observações que deseja:", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto1_obs.grid(column=0, row=0, padx=10, pady=10)

    n = tk.Text(frame2, height=1, width=10)
    n.grid(column=0, row=1, padx=10, pady=10)

    texto2_obs = tk.Label(frame2, text="Digite o intervalo entre duas observações:", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto2_obs.grid(column=0, row=2, padx=10, pady=10)

    t = tk.Text(frame2, height=1, width=10)
    t.grid(column=0, row=3, padx=10, pady=10)

    def guardar_valor_obs():
        v_n = int(n.get("1.0", "end-1c"))
        v_t = int(t.get("1.0", "end-1c"))
        N.append(v_n)
        T.append(v_t)

    frame3 = tk.Frame(FKE_obs, bg='#1E0127')
    frame3.pack()

    def limpar_valor_inicial():
        Q.clear()
        '''LEMBRAR DE LIMPAR AS MATRIZES P E R'''
        x_est.clear()

    voltar = tk.Button(frame3, text="Voltar", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=lambda: [FKE_obs.destroy(), limpar_valor_inicial()])
    voltar.grid(column=0, row=0, padx=10, pady=10)

    proximo = tk.Button(frame3, text="Próximo", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=lambda: [abrir_FKE_graficos(), guardar_valor_obs()])
    proximo.grid(column=1, row=0, padx=10, pady=10)

#Ângulos de Euler
def FKE_roll():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    plt.plot(Tempo, X_est_roll, color='red', label='Medida estimada')
    plt.plot(Tempo, x_real["roll"], linewidth=0.95, color='black', label='Medida real')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\phi$ (°)')
    plt.title('Ângulo $\\phi$ X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()
def FKE_pitch():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Pitch
    plt.plot(Tempo, X_est_pitch, color='lawngreen', label='Medida estimada')
    plt.plot(Tempo, x_real["pitch"], linewidth=0.95, color='black', label='Medida real')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\theta$ (°)')
    plt.title('Ângulo $\\theta$ X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()
def FKE_yaw():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Pitch
    plt.plot(Tempo, X_est_yaw, color='lawngreen', label='Medida estimada')
    plt.plot(Tempo, x_real["yaw"], linewidth=0.95, color='black', label='Medida real')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\theta$ (°)')
    plt.title('Ângulo $\\theta$ X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()

#Componentes do bias
def FKE_bias_x():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    #Bias Roll
    plt.plot(Tempo, x_real["bias_x"], linewidth=2.5, color='red', label='Medida real')
    plt.plot(Tempo, Bias_x, 'k:', label='Medida estimada')
    plt.xlabel('$t (h)$')
    plt.ylabel('$\\epsilon_{x}$ (°)')
    plt.title('Componente x do Bias X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()
def FKE_bias_y():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    #Bias Roll
    plt.plot(Tempo, x_real["bias_y"], linewidth=2.5, color='red', label='Medida real')
    plt.plot(Tempo, Bias_y, 'k:', label='Medida estimada')
    plt.xlabel('$t (h)$')
    plt.ylabel('$\\epsilon_{x}$ (°)')
    plt.title('Componente x do Bias X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()
def FKE_bias_z():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    #Bias Roll
    plt.plot(Tempo, x_real["bias_z"], linewidth=2.5, color='red', label='Medida real')
    plt.plot(Tempo, Bias_z, 'k:', label='Medida estimada')
    plt.xlabel('$t (h)$')
    plt.ylabel('$\\epsilon_{x}$ (°)')
    plt.title('Componente x do Bias X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()

#Covariâncias
def FKE_cov_roll():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Covariância Roll
    plt.plot(Tempo, Sigma1, color='tomato')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\epsilon_{\\phi}$')
    plt.title('Covariância de $\\phi$ X Tempo')
    plt.grid(True)
    plt.show()
def FKE_cov_pitch():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Covariância Roll
    plt.plot(Tempo, Sigma2, color='tomato')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\epsilon_{\\phi}$')
    plt.title('Covariância de $\\phi$ X Tempo')
    plt.grid(True)
    plt.show()
def FKE_cov_yaw():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Covariância Roll
    plt.plot(Tempo, Sigma3, color='tomato')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\epsilon_{\\phi}$')
    plt.title('Covariância de $\\phi$ X Tempo')
    plt.grid(True)
    plt.show()
def FKE_cov_bias_x():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Covariância Roll
    plt.plot(Tempo, Sigma4, color='tomato')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\epsilon_{\\phi}$')
    plt.title('Covariância de $\\phi$ X Tempo')
    plt.grid(True)
    plt.show()
def FKE_cov_bias_y():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Covariância Roll
    plt.plot(Tempo, Sigma5, color='tomato')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\epsilon_{\\phi}$')
    plt.title('Covariância de $\\phi$ X Tempo')
    plt.grid(True)
    plt.show()
def FKE_cov_bias_z():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    # Covariância Roll
    plt.plot(Tempo, Sigma6, color='tomato')
    plt.xlabel('$t (s)$')
    plt.ylabel('$\\epsilon_{\\phi}$')
    plt.title('Covariância de $\\phi$ X Tempo')
    plt.grid(True)
    plt.show()

#Resíduos dos sensores
def FKE_res_DSS1():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    #Resíduo e inovação - DSS 1
    plt.plot(Tempo, Inovação_1, color='maroon', label='Inovação 1')
    plt.plot(Tempo, Res_atual_1, color='royalblue', label='Resíduo atual 1')
    plt.xlabel('$t (s)$')
    plt.title('Inovação e resíduo DSS 1 X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()
def FKE_res_DSS2():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    #Resíduo e inovação - DSS 2
    plt.plot(Tempo, Inovação_2, color='purple', label='Inovação 2')
    plt.plot(Tempo, Res_atual_2, color='lightpink', label='Resíduo atual 2')
    plt.xlabel('$t (s)$')
    plt.title('Inovação e resíduo DSS 2 X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()
def FKE_res_IRES1():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    #Resíduo e inovação - IRES 1
    plt.plot(Tempo, Inovação_3, color='maroon', label='Inovação 1')
    plt.plot(Tempo, Res_atual_3, color='royalblue', label='Resíduo atual 1')
    plt.xlabel('$t (s)$')
    plt.title('Inovação e resíduo IRES 1 X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()
def FKE_res_IRES2():

    # Leitura da medida do giro
    global x_est
    Wgyro = pd.read_csv(r"omega_gyro.csv")

    # Leitura do vetor posição
    rr_leitura = pd.read_csv(r"orbit.csv")

    # Leitura de S0
    S0 = pd.read_csv(r"versor_sol.csv")

    # Leitura do x real
    x_real = pd.read_csv(r"x_real.csv")

    # Leitura do z real
    z_real = pd.read_csv(r"z_real.csv")

    # Intervalo de tempo
    dt = 0.5
    Tempo = [0]

    # Listas para gravar dados
    X_est_roll = [0]
    X_est_pitch = [0]
    X_est_yaw = [0]
    Bias_x = [0]
    Bias_y = [0]
    Bias_z = [0]
    Inovação_1 = [0]
    Inovação_2 = [0]
    Inovação_3 = [0]
    Inovação_4 = [0]
    Res_atual_1 = [0]
    Res_atual_2 = [0]
    Res_atual_3 = [0]
    Res_atual_4 = [0]
    Sigma1 = [0]
    Sigma2 = [0]
    Sigma3 = [0]
    Sigma4 = [0]
    Sigma5 = [0]
    Sigma6 = [0]

    for i in range(1, N[0], T[0]):

        ###----------------------- Propagação -----------------------###

        # Cálculo do vetor posição
        rr1 = rr_leitura["pos_x"][i]
        rr2 = rr_leitura["pos_y"][i]
        rr3 = rr_leitura["pos_z"][i]

        rr = np.sqrt(rr1 ** 2 + rr2 ** 2 + rr3 ** 2)

        # Velocidade angular orbital [rad/s]
        w0 = np.sqrt(3.986005e14 / (rr ** 3))  # movimento médio orbital (n^2=mi/r^3)

        # Propaga o estado (Runge Kutta 4):

        ne = 3

        West_x = Wgyro["Wgyro_x"][i] + x_est[3]
        West_y = Wgyro["Wgyro_y"][i] + x_est[4]
        West_z = Wgyro["Wgyro_z"][i] + x_est[5]

        West = np.array([[West_x, West_y, West_z]])

        # Função Runge-Kutta
        def din_system(w0, West, x_est):
            k = np.array([w0 * np.sin(x_est[2]) + West[0, 0] + x_est[1] * West[0, 2],
                          w0 * np.cos(x_est[2]) + West[0, 1] - x_est[0] * West[0, 2],
                          w0 * (x_est[1] * np.sin(x_est[2]) - x_est[0] * np.cos(x_est[2])) + West[0, 2] + x_est[0] *
                          West[0, 1]])
            return k

        # Calculando k1:
        k1 = din_system(w0, West, x_est)

        # Calculando k2:
        Wk2 = []
        for j in range(0, ne):
            wk2 = x_est[j] + 0.5 * dt * k1[j]
            Wk2.append(wk2)

        k2 = din_system(w0, West, Wk2)

        # Calculando k3:
        Wk3 = []
        for j in range(0, ne):
            wk3 = x_est[j] + 0.5 * dt * k2[j]
            Wk3.append(wk3)

        k3 = din_system(w0, West, Wk3)

        # Calculando k4:
        Wk4 = []
        for j in range(0, ne):
            wk4 = x_est[j] + dt * k3[j]
            Wk4.append(wk4)

        k4 = din_system(w0, West, Wk4)

        y = [0, 0, 0]

        for j in range(0, ne):
            rk = x_est[j] + dt * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6
            y[j] = rk

        x_prop = np.array([y[0], y[1], y[2], x_est[3], x_est[4], x_est[5]])

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT = np.array([x_prop[0], x_prop[1], x_prop[2]])
        S0_x = S0["S0_x"][i]
        S0_y = S0["S0_y"][i]
        S0_z = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi = AT[0]
        theta = AT[1]
        psi = AT[2]

        # Inicialização das derivadas parciais
        HPSIdss = [0, 0, 0, 0, 0, 0]
        HTETdss = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx = S0_x + psi * S0_y - theta * S0_z
        Sby = S0_y - psi * S0_x + phi * S0_z
        Sbz = S0_z - phi * S0_y + theta * S0_z

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi = Sbx * np.cos(np.deg2rad(60)) + Sbz * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi) >= np.cos(np.deg2rad(60)):
            ALFA_PSI = np.arctan(-Sby / aux_alfa_psi)

        else:  # medida inválida
            print('Medida alfa_psi é inválida pois aux_alfa_psi < cos(60)')
            ALFA_PSI = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET = np.deg2rad(24) - np.arctan(Sbx / Sbz)

        if abs(ALFA_TET) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet é inválida pois valor >= 60°')
            ALFA_TET = 9.99e99

        DSS = np.array([ALFA_PSI, ALFA_TET])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr = 0
        dsxdp = -S0_z
        dsxdy = S0_y

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr = S0_z
        dsydp = 0
        dsydy = -S0_x

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr = -S0_y
        dszdp = S0_x
        dszdy = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y = -((aux_alfa_psi) ** 2 + Sby ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p = -(Sbx ** 2 + Sbz ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1 = (dsydr * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdr + np.cos(np.deg2rad(150)) * dszdr)) / d_y
        hapsi2 = (dsydp * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdp + np.cos(np.deg2rad(150)) * dszdp)) / d_y
        hapsi3 = (dsydy * aux_alfa_psi - Sby * (np.cos(np.deg2rad(60)) * dsxdy + np.cos(np.deg2rad(150)) * dszdy)) / d_y

        hatet1 = (dsxdr * Sbz - Sbx * dszdr) / d_p
        hatet2 = (dsxdp * Sbz - Sbx * dszdp) / d_p
        hatet3 = (dsxdy * Sbz - Sbx * dszdy) / d_p

        HPSIdss = [hapsi1, hapsi2, hapsi3, 0, 0, 0]
        HTETdss = [hatet1, hatet2, hatet3, 0, 0, 0]

        H_DSS = np.array([[HPSIdss[0], HPSIdss[1], HPSIdss[2], HPSIdss[3], HPSIdss[4], HPSIdss[5]],
                          [HTETdss[0], HTETdss[1], HTETdss[2], HTETdss[3], HTETdss[4], HTETdss[5]]])

        # Sensores IRES
        DFIH = 2e-4
        DTETH = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires = [1, 0, 0, 0, 0, 0]
        HTETires = [0, 1, 0, 0, 0, 0]

        H_IRES = np.array([[HFIires[0], HFIires[1], HFIires[2], HFIires[3], HFIires[4], HFIires[5]],
                           [HTETires[0], HTETires[1], HTETires[2], HTETires[3], HTETires[4], HTETires[5]]])

        # Cálculo de roll e pitch
        FI = phi + DFIH
        TET = theta + DTETH

        IRES = np.array([FI, TET])

        ###----------------------- Atualização -----------------------###

        # Calculo das matrizes jacobianas F e H;
        f11 = 0
        f12 = West[0, 2]
        f13 = w0 * np.cos(np.rad2deg(AT[2]))
        f14 = 1
        f15 = 0
        f16 = AT[1]

        f21 = -West[0, 2]
        f22 = 0
        f23 = -w0 * np.sin(np.rad2deg(AT[2]))
        f24 = 0
        f25 = 1
        f26 = -AT[0]

        f31 = -w0 * np.cos(np.rad2deg(AT[2])) + West[0, 1]
        f32 = w0 * np.sin(np.rad2deg(AT[2]))
        f33 = w0 * (AT[1] * np.cos(np.rad2deg(AT[2])) + AT[0] * np.sin(np.rad2deg(AT[2])))
        f34 = 0
        f35 = AT[0]
        f36 = 1

        f41, f42, f43, f44, f45, f46 = 0, 0, 0, 0, 0, 0

        f51, f52, f53, f54, f55, f56 = 0, 0, 0, 0, 0, 0

        f61, f62, f63, f64, f65, f66 = 0, 0, 0, 0, 0, 0

        F = np.array([[f11, f12, f13, f14, f15, f16], [f21, f22, f23, f24, f25, f26],
                      [f31, f32, f33, f34, f35, f36], [f41, f42, f43, f44, f45, f46],
                      [f51, f52, f53, f54, f55, f56], [f61, f62, f63, f64, f65, f66]])

        # Matriz de transição (fi):
        fi = scipy.linalg.expm(dt * F)

        # Matriz de covariancia propagada:
        P_prop = fi * P_est * fi.T + Q

        # Atualização do filtro:
        H = np.array([[H_DSS[0, 0], H_DSS[0, 1], H_DSS[0, 2], H_DSS[0, 3], H_DSS[0, 4], H_DSS[0, 5]],
                      [H_DSS[1, 0], H_DSS[1, 1], H_DSS[1, 2], H_DSS[1, 3], H_DSS[1, 4], H_DSS[1, 5]],
                      [H_IRES[0, 0], H_IRES[0, 1], H_IRES[0, 2], H_IRES[0, 3], H_IRES[0, 4], H_IRES[0, 5]],
                      [H_IRES[1, 0], H_IRES[1, 1], H_IRES[1, 2], H_IRES[1, 3], H_IRES[1, 4], H_IRES[1, 5]]])

        res1 = z_real["DSS1"][i] - DSS[0]
        res2 = z_real["DSS2"][i] - DSS[1]
        res3 = z_real["IRES1"][i] - IRES[0]
        res4 = z_real["IRES2"][i] - IRES[1]

        res = np.array([res1, res2, res3, res4])

        Kaux = P_prop.dot(H.T)  # OBS: A partir daqui, valores inf e nan são encontrados
        K = Kaux.dot(np.linalg.inv(H.dot(Kaux) + Rk))  # Possível problema nos valores iniciais --> testar ajustes
        P_est1 = (identity(6) - K.dot(H)) * P_prop
        x_aux = x_prop + K.dot(res)
        x_est = x_aux

        # Desvio-Padrão da Covariância

        sigma1 = np.rad2deg(np.sqrt(P_est1[0, 0]))
        sigma2 = np.rad2deg(np.sqrt(P_est1[1, 1]))
        sigma3 = np.rad2deg(np.sqrt(P_est1[2, 2]))
        sigma4 = np.rad2deg(np.sqrt(P_est1[3, 3]))
        sigma5 = np.rad2deg(np.sqrt(P_est1[4, 4]))
        sigma6 = np.rad2deg(np.sqrt(P_est1[5, 5]))  # [deg/s]

        ###----------------------- Resíduos pós atualização -----------------------###

        ###----------------------- Sensores -----------------------###

        # Sensores DSS
        AT_est = np.array([x_est[0], x_est[1], x_est[2]])
        S0_x_est = S0["S0_x"][i]
        S0_y_est = S0["S0_y"][i]
        S0_z_est = S0["S0_z"][i]

        # Roll, Pitch, Yaw
        phi_est = AT_est[0]
        theta_est = AT_est[1]
        psi_est = AT_est[2]

        # Inicialização das derivadas parciais
        HPSIdss_est = [0, 0, 0, 0, 0, 0]
        HTETdss_est = [0, 0, 0, 0, 0, 0]

        # Vetor solar no sistema de coordenadas fixas no corpo do satélite
        Sbx_est = S0_x_est + psi_est * S0_y_est - theta_est * S0_z_est
        Sby_est = S0_y_est - psi_est * S0_x_est + phi_est * S0_z_est
        Sbz_est = S0_z_est - phi_est * S0_y_est + theta_est * S0_z_est

        # Teste para verificar se a medida de yaw é válida
        aux_alfa_psi_est = Sbx_est * np.cos(np.deg2rad(60)) + Sbz_est * np.cos(np.deg2rad(150))

        if abs(aux_alfa_psi_est) >= np.cos(np.deg2rad(60)):
            ALFA_PSI_est = np.arctan(-Sby_est / aux_alfa_psi_est)

        else:  # medida inválida
            print('Medida alfa_psi_est é inválida pois aux_alfa_psi_est < cos(60)')
            ALFA_PSI_est = 9.99e99

            # Teste para verificar se a medida de pitch é válida
        ALFA_TET_est = np.deg2rad(24) - np.arctan(Sbx_est / Sbz_est)

        if abs(ALFA_TET_est) >= np.deg2rad(60):  # medida inválida
            print('Medida alfa_tet_est é inválida pois valor >= 60°')
            ALFA_TET_est = 9.99e99

        DSS_est = np.array([ALFA_PSI_est, ALFA_TET_est])

        # Parcelas preliminares das derivadas parciais

        # dsxdr= dSbx/dr, com dr: derivada em roll
        dsxdr_est = 0
        dsxdp_est = -S0_z_est
        dsxdy_est = S0_y_est

        # dsydp= dSby/dp, com dp: derivada em pitch
        dsydr_est = S0_z_est
        dsydp_est = 0
        dsydy_est = -S0_x_est

        # dszdy= dSbz/dy, com dy: derivada em yaw
        dszdr_est = -S0_y_est
        dszdp_est = S0_x_est
        dszdy_est = 0

        # d_y = denominador da derivada parcial com relação à yaw
        d_y_est = -((aux_alfa_psi_est) ** 2 + Sby_est ** 2)

        # d_p = denominador da derivada parcial com relação à pitch
        d_p_est = -(Sbx_est ** 2 + Sbz_est ** 2)

        # Derivadas parciais com relação ao DSS:
        hapsi1_est = (dsydr_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdr_est + np.cos(np.deg2rad(150)) * dszdr_est)) / d_y_est
        hapsi2_est = (dsydp_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdp_est + np.cos(np.deg2rad(150)) * dszdp_est)) / d_y_est
        hapsi3_est = (dsydy_est * aux_alfa_psi_est - Sby_est * (np.cos(np.deg2rad(60)) * dsxdy_est + np.cos(np.deg2rad(150)) * dszdy_est)) / d_y_est

        hatet1_est = (dsxdr_est * Sbz_est - Sbx_est * dszdr_est) / d_p_est
        hatet2_est = (dsxdp_est * Sbz_est - Sbx_est * dszdp_est) / d_p_est
        hatet3_est = (dsxdy_est * Sbz_est - Sbx_est * dszdy_est) / d_p_est

        HPSIdss_est = [hapsi1_est, hapsi2_est, hapsi3_est, 0, 0, 0]
        HTETdss_est = [hatet1_est, hatet2_est, hatet3_est, 0, 0, 0]

        H_DSS_est = np.array([[HPSIdss_est[0], HPSIdss_est[1], HPSIdss_est[2], HPSIdss_est[3], HPSIdss_est[4], HPSIdss_est[5]], [HTETdss_est[0], HTETdss_est[1], HTETdss_est[2], HTETdss_est[3], HTETdss_est[4], HTETdss_est[5]]])

        # Sensores IRES
        DFIH_est = 2e-4
        DTETH_est = 1e-4

        # Matriz de Observação com relação ao IRES associada as componentes de roll e pitch
        HFIires_est = [1, 0, 0, 0, 0, 0]
        HTETires_est = [0, 1, 0, 0, 0, 0]

        H_IRES_est = np.array([[HFIires_est[0], HFIires_est[1], HFIires_est[2], HFIires_est[3], HFIires_est[4], HFIires_est[5]], [HTETires_est[0], HTETires_est[1], HTETires_est[2], HTETires_est[3], HTETires_est[4], HTETires_est[5]]])

        # Cálculo de roll e pitch
        FI_est = phi_est + DFIH_est
        TET_est = theta_est + DTETH_est

        IRES_est = np.array([FI_est, TET_est])

        z_atual = np.array([[DSS_est[0]], [DSS_est[1]], [IRES_est[0]], [IRES_est[1]]]).T

        res1_atual = z_real["DSS1"][i] - z_atual[0, 0]
        res2_atual = z_real["DSS2"][i] - z_atual[0, 1]
        res3_atual = z_real["IRES1"][i] - z_atual[0, 2]
        res4_atual = z_real["IRES2"][i] - z_atual[0, 3]

        res_atual = np.array([[res1_atual], [res2_atual], [res3_atual], [res4_atual]])

        # Passando para graus
        x_estg = np.array([np.rad2deg(x_est[0]), np.rad2deg(x_est[1]), np.rad2deg(x_est[2]), np.rad2deg(x_est[3] * 3600), np.rad2deg(x_est[4] * 3600), np.rad2deg(x_est[5] * 3600)])  # [deg deg/s]

        inov = np.array([np.rad2deg(res[0]), np.rad2deg(res[1]), np.rad2deg(res[2]), np.rad2deg(res[3])])  # resíduo pós-propagação (inovação)

        residuo_atual = np.array([np.rad2deg(res_atual[0]), np.rad2deg(res_atual[1]), np.rad2deg(res_atual[2]), np.rad2deg(res_atual[3])])  # resíduo pós atualização

        # Armazenar os dados
        X_est_roll.append(x_estg[0])
        X_est_pitch.append(x_estg[1])
        X_est_yaw.append(x_estg[2])
        Bias_x.append(x_estg[3])
        Bias_y.append(x_estg[4])
        Bias_z.append(x_estg[5])
        Inovação_1.append(inov[0])
        Inovação_2.append(inov[1])
        Inovação_3.append(inov[2])
        Inovação_4.append(inov[3])
        Res_atual_1.append(residuo_atual[0])
        Res_atual_2.append(residuo_atual[1])
        Res_atual_3.append(residuo_atual[2])
        Res_atual_4.append(residuo_atual[3])
        Sigma1.append(sigma1)
        Sigma2.append(sigma2)
        Sigma3.append(sigma3)
        Sigma4.append(sigma4)
        Sigma5.append(sigma5)
        Sigma6.append(sigma6)
        Tempo.append(i)

    #Resíduo e inovação - IRES 2
    plt.plot(Tempo, Inovação_4, color='purple', label='Inovação 2')
    plt.plot(Tempo, Res_atual_4, color='lightpink', label='Resíduo atual 2')
    plt.xlabel('$t (s)$')
    plt.title('Inovação e resíduo IRES 2 X Tempo')
    plt.grid(True)
    plt.legend()
    plt.show()

def abrir_FKE_graficos():
    FKE_graf = tk.Toplevel()
    FKE_graf.title("FKE - Gráficos")
    FKE_graf.config(background='#1E0127')
    FKE_graf.focus_force()
    FKE_graf.grab_set()

    frame1 = tk.Frame(FKE_graf, bg='#1E0127')
    frame1.pack()

    topico1_gra = tk.Label(frame1, text="Gráficos", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico1_gra.grid(column=0, row=0, padx=10, pady=10)

    texto1_gra = tk.Label(frame1, text="Para vizualizar o gráfico em uma nova janela, selecione abaixo o gráfico que deseja:", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto1_gra.grid(column=0, row=1, padx=10, pady=10)

    frame2 = tk.Frame(FKE_graf, bg='#1E0127')
    frame2.pack()

    texto2_gra = tk.Label(frame2, text="Componentes do vetor de estado", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto2_gra.grid(column=0, row=0, padx=10, pady=10)

    #Gráficos das componentes do vetor de estado
    roll = tk.Button(frame2, text="Ângulo roll", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_roll)
    roll.grid(column=0, row=1, padx=10, pady=10)
    pitch = tk.Button(frame2, text="Ângulo pitch", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_pitch)
    pitch.grid(column=0, row=2, padx=10, pady=10)
    yaw = tk.Button(frame2, text="Ângulo yaw", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_yaw)
    yaw.grid(column=0, row=3, padx=10, pady=10)
    bias_x = tk.Button(frame2, text="Componente x do bias", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_bias_x)
    bias_x.grid(column=0, row=4, padx=10, pady=10)
    bias_y = tk.Button(frame2, text="Componente y do bias", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_bias_y)
    bias_y.grid(column=0, row=5, padx=10, pady=10)
    bias_z = tk.Button(frame2, text="Componente z do bias", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_bias_z)
    bias_z.grid(column=0, row=6, padx=10, pady=10)

    texto3_gra = tk.Label(frame2, text="Covariâncias do vetor de estado", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto3_gra.grid(column=1, row=0, padx=10, pady=10)

    # Gráficos das covariâncias
    cov_roll = tk.Button(frame2, text="Covariância - roll", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_cov_roll)
    cov_roll.grid(column=1, row=1, padx=10, pady=10)
    cov_pitch = tk.Button(frame2, text="Covariância - pitch", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_cov_pitch)
    cov_pitch.grid(column=1, row=2, padx=10, pady=10)
    cov_yaw = tk.Button(frame2, text="Covariância - yaw", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_cov_yaw)
    cov_yaw.grid(column=1, row=3, padx=10, pady=10)
    cov_bias_x = tk.Button(frame2, text="Covariância - x do bias", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_cov_bias_x)
    cov_bias_x.grid(column=1, row=4, padx=10, pady=10)
    cov_bias_y = tk.Button(frame2, text="Covariância - y do bias", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_cov_bias_y)
    cov_bias_y.grid(column=1, row=5, padx=10, pady=10)
    cov_bias_z = tk.Button(frame2, text="Covariância - z do bias", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_cov_bias_z)
    cov_bias_z.grid(column=1, row=6, padx=10, pady=10)

    texto4_gra = tk.Label(frame2, text="Covariâncias do vetor de estado", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto4_gra.grid(column=2, row=0, padx=10, pady=10)

    # Gráficos dos resíduos e inovações
    resDSS1 = tk.Button(frame2, text="DSS 1", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_res_DSS1)
    resDSS1.grid(column=2, row=1, padx=10, pady=10)
    resDSS2 = tk.Button(frame2, text="DSS 2", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_res_DSS2)
    resDSS2.grid(column=2, row=2, padx=10, pady=10)
    resIRES1 = tk.Button(frame2, text="IRES 1", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_res_IRES1)
    resIRES1.grid(column=2, row=3, padx=10, pady=10)
    resIRES2 = tk.Button(frame2, text="IRES 2", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=FKE_res_IRES2)
    resIRES2.grid(column=2, row=4, padx=10, pady=10)

    def limpar_valor_obs():
        N.clear()
        T.clear()

    voltar = tk.Button(frame2, text="Voltar", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=lambda: [FKE_graf.destroy(), limpar_valor_obs()])
    voltar.grid(column=2, row=6, padx=10, pady=10)

def abrir_janela_doc():
    menu.withdraw()
    janela_doc = tk.Toplevel()
    janela_doc.title("Documentos")
    janela_doc.config(background='#1E0127')
    janela_doc.focus_force()
    janela_doc.grab_set()

    '''
    capa_doc = tk.PhotoImage(file="capa_ref.png")
    colocar_capa_doc = tk.Label(janela_doc, image=capa_doc, background='white')
    colocar_capa_doc.pack(padx=10, pady=10)
    '''

    topico1_doc = tk.Label(janela_doc, text="Documentos e manual de instruções", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico1_doc.pack(padx=10, pady=10)

    texto1_doc = tk.Label(janela_doc, text="Selecione abaixo o que deseja:", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto1_doc.pack(padx=10, pady=10)

    lista1_doc = ['Link para relatório base do projeto', 'Link para manual de instruções', 'Link para download do código']
    cb_doc = ttk.Combobox(janela_doc, values=lista1_doc, width=30)
    cb_doc.pack(padx=10, pady=10)

    def printar_resp_cb_doc():
        resp_cb_doc = cb_doc.get()
        if resp_cb_doc == lista1_doc[0]:
            resultado_cb_doc["text"] = 'https://drive.google.com/file/d/1bSUrXKRMVP6FwsWc31jTKHAJfxJKRA2O/view?usp=sharing'
        elif resp_cb_doc == lista1_doc[1]:
            resultado_cb_doc["text"] = 'https://drive.google.com/file/d/1Szn9UeumgoSIJh216CRKfbPfxPJSwnG0/view?usp=sharing'
        elif resp_cb_doc == lista1_doc[2]:
            resultado_cb_doc["text"] = 'https://drive.google.com/file/d/1jOPlf9J_iEU5NJbvterNl8IoFv7S4_Nm/view?usp=sharing'
        else:
            resultado_cb_doc["text"] = 'Selecione algum item'

    resultado_cb_doc = tk.Label(janela_doc, text="", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    resultado_cb_doc.pack(padx=10, pady=10)

    botao_resp_cb_doc = tk.Button(janela_doc, text="Selecionar", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=printar_resp_cb_doc)
    botao_resp_cb_doc.pack(padx=10, pady=10)

    botao_voltar_menu = tk.Button(janela_doc, text="Voltar para Menu Inicial", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=lambda: [janela_doc.destroy(), menu.deiconify()])
    botao_voltar_menu.pack(padx=10, pady=10)

def abrir_janela_contatos():
    janela_contatos = tk.Toplevel()
    janela_contatos.title("Entre em contato")
    janela_contatos.config(background='#1E0127')
    janela_contatos.focus_force()
    janela_contatos.grab_set()

    topico1_contatos = tk.Label(janela_contatos, text="Contatos", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico1_contatos.pack(padx=10, pady=10)

    texto1_contatos = tk.Label(janela_contatos, text="Caso tenha alguma dúvida ou sugestão, entre em contato \n"
                                                      "através dos meios disponibilizados abaixo:", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto1_contatos.pack(padx=10, pady=10)

    # Criar a borda
    frame_fundo = tk.Frame(janela_contatos, bg='#1E0127')
    frame_fundo.pack()
    frame_borda = tk.Frame(frame_fundo, bg='#FFDE93')
    frame_borda.pack(padx=5, pady=5)

    frame2 = tk.Frame(frame_borda, bg='#1E0127')
    frame2.pack(padx=1, pady=1)

    texto2_contatos = tk.Label(frame2, text="Instagram", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    texto2_contatos.pack(padx=10, pady=10)

    texto3_contatos = tk.Label(janela_contatos, text="@bru020", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto3_contatos.pack(padx=10, pady=10)

    # Criar a borda
    frame_fundo = tk.Frame(janela_contatos, bg='#1E0127')
    frame_fundo.pack()
    frame_borda = tk.Frame(frame_fundo, bg='#FFDE93')
    frame_borda.pack(padx=5, pady=5)

    frame3 = tk.Frame(frame_borda, bg='#1E0127')
    frame3.pack(padx=1, pady=1)

    texto4_contatos = tk.Label(frame3, text="Linkedin", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    texto4_contatos.pack(padx=10, pady=10)

    texto5_contatos = tk.Label(janela_contatos, text="linkedin.com/in/bruno-gomes-cordeiro-b35697240/", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto5_contatos.pack(padx=10, pady=10)

    # Criar a borda
    frame_fundo = tk.Frame(janela_contatos, bg='#1E0127')
    frame_fundo.pack()
    frame_borda = tk.Frame(frame_fundo, bg='#FFDE93')
    frame_borda.pack(padx=5, pady=5)

    frame4 = tk.Frame(frame_borda, bg='#1E0127')
    frame4.pack(padx=1, pady=1)

    texto6_contatos = tk.Label(frame4, text="E-mail", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    texto6_contatos.pack(padx=10, pady=10)

    texto7_contatos = tk.Label(janela_contatos, text="bru020@usp.br", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto7_contatos.pack(padx=10, pady=10)

    botao_voltar_menu = tk.Button(janela_contatos, text="Voltar para Menu Inicial", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=janela_contatos.destroy)
    botao_voltar_menu.pack(padx=10, pady=10)

def abrir_janela_infos():
    menu.withdraw()
    janela_info = tk.Toplevel()
    janela_info.title("Sobre o FKE Software")
    janela_info.config(background='#1E0127')
    janela_info.focus_force()
    janela_info.grab_set()

    '''
    capa_info = tk.PhotoImage(file="capa_info.png")
    colocar_capa_info = tk.Label(janela_info, image=capa_info, background='#1E0127')
    colocar_capa_info.pack(padx=10, pady=10)
    '''

    topico1_info = tk.Label(janela_info, text="O que é o FKE?", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico1_info.pack(padx=10, pady=10)

    texto1_info = tk.Label(janela_info, text="O Filtro de Kalman Estendido (FKE) é uma ferramenta com o intuito de \n "
                                             "estimar algumas varíaveis afim de prever possíveis erros e ruídos no sistema \n"
                                             "considerado. Neste software, o FKE foi utilizado para calcular o vetor de \n"
                                             "estado, ruídos e covariâncias de um satélite em órbita, com o objetivo de \n"
                                             "supervisionar sua atitude (orientação do satélite).", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto1_info.pack(padx=10, pady=10)

    topico2_info = tk.Label(janela_info, text="O que é este software?", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico2_info.pack(padx=10, pady=10)

    texto2_info = tk.Label(janela_info, text="Este software foi desenvolvido em Python com o objetivo de facilitar \n"
                                             "a utilização do FKE, uma vez que grande parte dos códigos são disponibilziados \n"
                                             "apenas em Matlab. Ademais, também conta com uma interface, que ajuda \n"
                                             "na busca dos resultados.",
                           background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto2_info.pack(padx=10, pady=10)

    topico2_info = tk.Label(janela_info, text="Como usar?", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
    topico2_info.pack(padx=10, pady=10)

    texto3_info = tk.Label(janela_info, text="Para usar este software, basta clicar em 'Ir para o FKE' no Menu \n"
                                             "e informar os dados utilizando os botões nas telas posteriores. \n"
                                             "Ressalta-se que este código não funciona para todos os casos possíveis \n"
                                             "de utilização do FKE, uma vez que foi baseado em uma pesquisa específica \n"
                                             "para o satélite CBERS. Em 'Referências', é possível acessar o trabalho completo.",
                           background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
    texto3_info.pack(padx=10, pady=10)

    botao_voltar_menu = tk.Button(janela_info, text="Voltar para Menu Inicial", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=lambda: [janela_info.destroy(), menu.deiconify()])
    botao_voltar_menu.pack(padx=10, pady=10)

menu = tk.Tk() #Abrir menu

menu.title('Menu Inicial')
menu.config(background='#1E0127')

texto1_menu = tk.Label(menu, text="Seja bem-vindo(a) ao FKE Software", background='#1E0127', foreground='#FFDE93', font=("Times New Roman", 15))
texto1_menu.grid(column=0, row=0, padx=10, pady=10)

icone = tk.PhotoImage(file="FKE_icone.png")
menu.iconphoto(True, icone)
logo = tk.PhotoImage(file="FKE_logo_cort.png")
colocar_logo = tk.Label(menu, image=logo, background='#1E0127')
colocar_logo.grid(column=0, row=1, padx=10, pady=10)

texto2_menu = tk.Label(menu, text="Para prosseguir, escolha uma das opções abaixo:", background='#1E0127', foreground='#E0FFFF', font=("Arial", 12))
texto2_menu.grid(column=0, row=2, padx=10, pady=10)

botao_infos = tk.Button(menu, text="Sobre o FKE Software", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=abrir_janela_infos)
botao_infos.grid(column=0, row=3, padx=10, pady=10)

botao_FKE = tk.Button(menu, text="Ir para o FKE", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=abrir_FKE_cond_iniciais)
botao_FKE.grid(column=0, row=4, padx=10, pady=10)

botao_contatos = tk.Button(menu, text="Entre em contato", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=abrir_janela_contatos)
botao_contatos.grid(column=0, row=5, padx=10, pady=10)

botao_ref = tk.Button(menu, text="Documentos", background='#DDA0DD', foreground='#1E0127', font=("Arial", 12), height=1, width=20, command=abrir_janela_doc)
botao_ref.grid(column=0, row=6, padx=10, pady=10)

menu.mainloop()