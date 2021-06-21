#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TOV import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst
import scipy.optimize
import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, splrep

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib.transforms import Bbox


# In[2]:


def run(opti,rho_cen):
    PhiInit = 1
    PsiInit = 0
    option = opti
    radiusMax_in = 40000
    radiusMax_out = 10000000
    Npoint = 1000000

    log_active = True
    dilaton_active = True
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active)
    tov.ComputeTOV()
    # tov.Plot()
    r = tov.radius
    a = tov.g_tt
    b = tov.g_rr
    phi = tov.Phi
    phi_dot = tov.Psi
    radiusStar = tov.radiusStar
    mass_ADM = tov.massADM / (1.989*10**30) # in solar mass
    a_dot = (-a[1:-2]+a[2:-1])/(r[2:-1]-r[1:-2])
    b_dot = (-b[1:-2]+b[2:-1])/(r[2:-1]-r[1:-2])
    f_a = -a_dot*r[1:-2]*r[1:-2]/1000
    f_b = -b_dot*r[1:-2]*r[1:-2]/1000
    f_phi = -phi_dot*r*r/1000

    print('f_a at infinity ', f_a[-1])
    print('f_b at infinity ', f_b[-1])
    print('f_phi at infinity ', f_phi[-1])
    if option == 1:
        b_ = 1/(2*np.sqrt(3))
        C = f_b[-1]/f_phi[-1]
        a1 = 1
        a2 = 4*b_*(C+1)
        a3 = -1
        a4 = a2*a2-4*a1*a3
        gamma = (-a2-np.sqrt(a4))/(2*a1)
        r_m = 1000*f_phi[-1]*(1+gamma**2)/(4*b_*gamma)
        comment = '$L_m = -\\rho$'
        descr = 'rho'
    elif option == 2:
        b_ = -1/(2*np.sqrt(3))
        C = f_b[-1]/f_phi[-1]
        a1 = 1
        a2 = 4*b_*(C+1)
        a3 = -1
        a4 = a2*a2-4*a1*a3
        gamma = (-a2-np.sqrt(a4))/(2*a1)
        r_m = 1000*f_phi[-1]*(1+gamma**2)/(4*b_*gamma)
        comment = '$L_m = P$'
        descr = 'P'
    else:
        print('not a valid option, try 1 or 2')
    print('gamma', gamma)
    print('r_m',r_m )
    r_2 = np.linspace(r_m,r[-1],num=100000)
    exp_phi = -4*b_*gamma/(1+gamma**2)
    exp_rho = (gamma**2)/(1+gamma**2)-exp_phi/2
    exp_a = (1-gamma**2)/(1+gamma**2)-exp_phi
    exp_b = -(1-gamma**2)/(1+gamma**2)-exp_phi
    rho_u = r_2*(1-r_m/r_2)**exp_rho
    a_u = (1-r_m/r_2)**exp_a
    phi_u= (1-r_m/r_2)**exp_phi
    drho_dr_u = (1-r_m/r_2)**exp_rho+exp_rho*(r_m/r_2)*(1-r_m/r_2)**(exp_rho-1)
    b_u = ((1-r_m/r_2)**exp_b)*(drho_dr_u)**(-2)
    r_lim = 80000
    if option == 1:
        couleur = (0.85,0.325,0.098)
        nom = '$\\alpha_-$'
        alphag = gamma
    elif option == 2:
        couleur = (0.929,0.694,0.125)
        nom = '$\\alpha_+$'
        alphag = - gamma

    return (r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,alphag)


# In[3]:


def make_plots(option,r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,gamma):


    a_interpol = interp1d(rho_u, a_u, fill_value="extrapolate")
    diff_a = (a_interpol(r)-a) / a


    b_interpol = interp1d(rho_u, b_u, fill_value="extrapolate")
    diff_b = (b_interpol(r)-b) / b

    ab_interpol = interp1d(rho_u, a_u * b_u, fill_value="extrapolate")
    diff_ab = (ab_interpol(r)- a * b) / (a * b)

    phi_interpol = interp1d(rho_u, phi_u, fill_value="extrapolate")
    diff_phi = (phi_interpol(r)-phi) / phi


    plt.plot(r*1e-3,a*b,label=f'a*b: numerical ({comment})',color=(0.,0.447,0.741))
    plt.plot(rho_u*1e-3,a_u*b_u,linestyle='dashed',label = f'a*b: {nom}',color = couleur)
    plt.axvline(x=radiusStar*1e-3, color='black')
    plt.axhline(y= 1, color = 'gray')
    plt.xlim([0,r_lim*1e-3])
    plt.ylim([a[0]*b[0],max(a*b)+0.1])
    plt.xlabel('Radius r (km)')
    plt.ylabel('a * b', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/ab_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()


    plt.plot(r*1e-3,phi,label=f'$\\Phi$: numerical({comment})',color=(0.,0.447,0.741))
    plt.plot(rho_u*1e-3,phi_u,linestyle='dashed',label= f'$\\Phi$: {nom}',color= couleur)
    plt.axvline(x=radiusStar*1e-3, color='black')
    plt.xlim([0,r_lim*1e-3])
    if option == 1:
        limitaY = [phi[0],1.01]
    if option == 2:
        limitaY = [1,1.1]
    plt.ylim(limitaY)
    plt.xlabel('Radius r (km)')
    plt.ylabel('$\\Phi$', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/phi_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()

    
    diff_a_wo_offset = diff_a - diff_a[-1]  
    diff_b_wo_offset = diff_b - diff_b[-1]  
    diff_phi_wo_offset = diff_phi - diff_phi[-1]
    
    plt.plot(r*1e-3,diff_a_wo_offset*100,label=f'$a$ (numerical({comment}) - {nom}) in %\n(Offset = {diff_a[-1]*100:.2f}%)',color=(0.85,0.325,0.098))
#     plt.plot(r*1e-3,diff_b_wo_offset*100,label=f'$b$ (numerical({comment}) - {nom}) in %\n(Offset = {diff_b[-1]*100:.2f}\%)',color=(0.,0.447,0.741))
    plt.plot(r*1e-3,diff_b*100,label=f'$b$ (numerical({comment}) - {nom}) in %',color=(0.,0.447,0.741), linestyle='dashed')
#     plt.plot(r*1e-3,diff_phi*100,label=f'$\\Phi$ (numerical({comment}) - {nom}) in %',color=(0.929,0.694,0.125), linestyle='dashdot')
    plt.plot(r*1e-3,diff_phi_wo_offset*100,label=f'$\\Phi$ (numerical({comment}) - {nom}) in %\n(Offset = {diff_phi[-1]*100:.2f}%)',color=(0.929,0.694,0.125), linestyle='dashdot')
    plt.axvline(x=radiusStar*1e-3, color='black')
    plt.axhline(y= 0, color = 'gray')
    plt.xlim([0,r_lim*1e-3])
    plt.ylim([-0.1,0.1])
    plt.xlabel('Radius r (km)')
    plt.ylabel('(numerical - analytical) %', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.legend()
    plt.savefig(f'figures/diff_awo_b_phi_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200,bbox_inches='tight')
    plt.show()


# In[4]:


for i in [1,2]:
#     for k in [100,500,1000,2000,4000,8000]:
    for k in [1000]:
        r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen, gamma  = run(i,k)
        make_plots(i,r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,k,gamma)


# In[5]:


def diff_at_radius(density,opti):

    central_density = density
    option = opti

    r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,gamma  = run(option,central_density)

    ab = a * b

    a_interpol = interp1d(rho_u, a_u, fill_value="extrapolate")
    b_interpol = interp1d(rho_u, b_u, fill_value="extrapolate")
    ab_interpol = interp1d(rho_u, a_u * b_u, fill_value="extrapolate")

    phi_interpol = interp1d(rho_u, phi_u, fill_value="extrapolate")



    diff_a = (a_interpol(r)*a[-1]/a_interpol(r)[-1]-a) / a 
    diff_b = (b_interpol(r)*b[-1]/b_interpol(r)[-1]-b) / b
    diff_ab = (ab_interpol(r)*ab[-1]/ab_interpol(r)[-1]-ab) / ab

    diff_phi = (phi_interpol(r)*phi[-1]/phi_interpol(r)[-1]-phi) / phi 

    i_radius = min(range(len(r)), key=lambda i: abs(r[i]-radiusStar))

    R_diff_a = diff_a[i_radius]
    R_diff_b = diff_b[i_radius]
    R_diff_ab = diff_ab[i_radius]

    R_diff_phi = diff_phi[i_radius]
    
    return R_diff_a,R_diff_b, R_diff_ab, R_diff_phi


# In[6]:


def make_gamma_beta_plots(n,opti):
    nspace = n
    option = opti
    
    if opti == 1:
        descr = 'rho'
        comment = '$L_m = - \rho$'
    elif opti == 2:
        descr = 'P'
        comment ='$L_m = P$'
    else:
        print('not a valid option, try 1 or 2')
    
    den_space = np.linspace(100,8000,num=n)
    
    gamma_vec = np.array([])
    beta_vec = np.array([])
    
    for den in den_space:
        r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,gamma  = run(option,den)
        gamma_vec = np.append(gamma_vec,gamma)
        beta_vec = np.append(beta_vec,(1-gamma**2)/(1+gamma**2))
    
    
    fig,ax = plt.subplots()
    plt.xlabel('Cetral density ($MeV/fm^3$)')
    ax.plot(den_space,gamma_vec, label=f'$\\alpha$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$\\alpha$', fontsize=12, color = 'tab:blue')
    
    ax2=ax.twinx()
    ax2.plot(den_space,beta_vec, label=f'$\\beta$ ({comment})', color = 'tab:green', linestyle = 'dashed')
    ax2.set_ylabel('$\\beta$', fontsize=12, color = 'tab:green')
    
    fig.legend(loc=0, bbox_to_anchor=(0.9, 0.7))
    plt.savefig(f'figures/alpha_beta_{nspace}_{descr}.png', dpi = 200,bbox_inches='tight')
    plt.show()


# In[7]:


make_gamma_beta_plots(50,1)
make_gamma_beta_plots(50,2)


# In[8]:


# Geodesics integration


# In[9]:


from scipy.integrate import solve_ivp


# In[16]:


# Definitions of the various functions used in the geodesics integration

def rho_2(r, m, beta): # rho^2
    return r**2*(1-2*m/(beta*r))**(1-beta)

def lambda_2(r, m, beta): #lambda^2
    return (1-2*m/(beta*r))**beta

def V_eff(r, m, beta, epsilon, L):
    return lambda_2(r, m, beta)*(epsilon+L**2/rho_2(r, m, beta))

def dr_dtau(r, E, m, beta, epsilon, L):
    return np.sqrt(E-V_eff(r, m, beta, epsilon, L))

def f_r(u): # u := dr/dtau
    return u

def f_u(r, u, m, beta, epsilon, L): # du/dtau
    F = 1-2*m/(beta*r)
    A = epsilon*m*F**(beta-1)
    B = ((L**2)/(beta*r**2))*(2*beta*m-beta*r+m)*F**(2*beta-2)
    return -(1/r**2)*(A+B)

def f_w(r, w, m, beta, epsilon, L, E): #dw/dt
    F = 1-2*m/(beta*r)
    return 2*m / F * (w / r)**2 + F**(2*beta) / E**2 * f_u(r, w, m, beta, epsilon, L)


def f_psi(r, m, beta, L): # dpsi/dtau
    return (L/r**2)*(1-2*m/(beta*r))**(beta-1)

def f_psi_w(r, m, beta, L, E): # dpsi/dt
    F = 1-2*m/(beta*r)
    dtau_dt = F**(beta) / E
    return f_psi(r, m, beta, L) * dtau_dt

# def dy_dtau(x, y, E, m, beta, epsilon, L):
#     r, u, psi = y
#     return [-f_r(u), -f_u(r, u, m, beta, epsilon, L), -f_psi(r, m, beta, L)]

def dy_dt(x, y, E, m, beta, epsilon, L):
    r, w, psi = y
    return [-f_r(w), -f_w(r, w, m, beta, epsilon, L, E), -f_psi_w(r, m, beta, L,E)]


# In[17]:


# Geodesics integration

def hit_singularity(t,r,m,beta): # Function to test if the geodesics hits the singularity
    return r[0] - 2 * m / abs(beta) - 0.5 # Ideally, should be r[0] - 2 * m / beta, but we added a safety gap of 0.5 for numerical purpose


def compute_geo_t(L, E, m, beta, epsilon, y0, tau_min, tau_max):
    tau = np.linspace(tau_min, tau_max, 500000)
    hit_event = lambda tau,x:hit_singularity(tau,x,m,beta)
    hit_event.terminal = True # Stop the numerical integration if close to singularity
    sol = solve_ivp(fun=lambda tau, y:dy_dt(tau, y, E, m, beta, epsilon, L), t_span=[tau_min, tau_max], y0=y0, method='BDF',events=hit_event, t_eval=tau)
    r = sol.y[0]
    psi = sol.y[2]
    x = r*np.cos(psi)
    y = r*np.sin(psi)
    return x, y


# In[18]:


def plot_one_geodesic_t(E,b,beta):
    #input
#     m = 2*abs(beta)/2 #<- mass
    m = 1  # Far away, same geodesics in both cases with this def
    tau_max = 10000 #<- affine parameter max bound
    tau_min = 0 #<- affine parameter min bound
    r0 = 84 #<- initial radius r(tau_min)
    epsilon = 1 #<- 0 massless/ 1 massive
    lim = 1.2*r0
    rs_ER = 2*m/abs(beta)
    rs = 2*m 

    L = E*b


    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')

    circle_s = plt.Circle((0,0),rs,color='k',fill=False)
    circle_s_ER = plt.Circle((0,0),rs_ER,color='grey',fill=False)

    ax.add_artist(circle_s)
    ax.add_artist(circle_s_ER)
    
    F = 1-2*m/(beta*r0)
    dtau_dt = F**(beta) / E

    y0 = [r0, dr_dtau(r0, E, m, beta, epsilon, L) * dtau_dt, -b/r0 * dtau_dt]
    x, y = compute_geo_t(L, E, m, beta, epsilon, y0, tau_min, tau_max)
    
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    
    plt.plot(x,y,color=(0.,0.447,0.741),label=f'ER (beta={beta})')
    x, y = compute_geo_t(L, E, m, 1, epsilon, y0, tau_min, tau_max)
    plt.plot(x,y,color=(0.85,0.325,0.098),label='GR',linestyle=':')
    plt.xlabel(r'$x=r\cos\psi$ (c=G=1)')
    plt.ylabel(r'$y=r\sin\psi$ (c=G=1)')
    plt.legend(loc=3)
    plt.savefig(f'figures/timelike_orbits_{E}_{b}_{beta}_t.png', dpi = 200,bbox_inches='tight')
    plt.show()


# In[19]:


def plot_multiple_lightray_t(beta):
    # plot light deformation -------------------------------------------------------
    #input
    E = 1 # with this convention L=b
    m = 1  # Far away, same geodesics in both cases with this def
    tau_max = 10000 #<- affine parameter max bound
    tau_min = 0 #<- affine parameter min bound
    r0 = 200 #<- initial radius r(tau_min)
    
    F = 1-2*m/(beta*r0)
    
    epsilon = 0 #<- 0 massless/ 1 massive
    rs = 2 * m
    rs_ER = 2*m/abs(beta)
    figure, axes = plt.subplots()
    axes.set_aspect(1)
    lim = 50
    axes.set_xlim([-lim,lim])
    axes.set_ylim([-lim,lim])
    circle_s = plt.Circle((0,0),rs,color='k',fill=False,label='Schwarzschild radius')
    circle_s_ER = plt.Circle((0,0),rs_ER,color='grey',fill=False, label='Naked singularity')
    axes.add_artist(circle_s)
    axes.add_artist(circle_s_ER)
    
    dtau_dt = F**(beta) / E
    
    for L in np.arange(10,0.01,-0.5):
        
        y0 = [r0, dr_dtau(r0, E, m, beta, epsilon, L) * dtau_dt, -L/r0 * dtau_dt]
        x, y = compute_geo_t(L, E, m, beta, epsilon, y0, tau_min, tau_max)
        r = np.sqrt(x**2+y**2)
        
        y0_GR = [r0, dr_dtau(r0, E, m, 1, epsilon, L), -L/r0]
        x_GR, y_GR = compute_geo_t(L, E, m, 1, epsilon, y0, tau_min, tau_max)
        r_GR = np.sqrt(x_GR**2+y_GR**2)


        axes.plot(x,y,color=(0.,0.447,0.741), label = f'ER (beta={beta})')
        axes.plot(x_GR,y_GR,color=(0.85,0.325,0.098),linestyle=':', label = 'GR')
    handles, labels = axes.get_legend_handles_labels()
    axes.legend([handles[0],handles[1]], [labels[0],labels[1]], loc = 3)
    plt.xlabel(r'$x=r\cos\psi$ (c=G=1)')
    plt.ylabel(r'$y=r\sin\psi$ (c=G=1)')
    plt.savefig(f'figures/null_orbits_{beta}_t.png', dpi = 200,bbox_inches='tight')
    plt.show()


# In[20]:


plot_multiple_lightray_t(0.88)


# In[21]:


plot_one_geodesic_t(0.98,4.5/0.98,0.88)


# In[ ]:




