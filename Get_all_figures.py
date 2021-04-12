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
        gamma = 1/(5+4*f_a[-1]/f_phi[-1])**(1/2)
        r_m = f_a[-1]*1000*(1+gamma**2)/(-1+5*gamma**2)
        comment = '$L_m = -\\rho$'
        descr = 'rho'
    elif option == 2:
        gamma = 1/(-3-4*f_a[-1]/f_phi[-1])**(1/2)
        r_m = -f_a[-1]*1000*(1+gamma**2)/(1+3*gamma**2)
        comment = '$L_m = P$'
        descr = 'P'
    else:
        print('not a valid option, try 1 or 2')
    print('gamma', gamma)
    print('r_m',r_m )
    r_2 = np.linspace(r_m,r[-1],num=100000)
    rho_m = ((r_2**2)*(1-(r_m/r_2))**((-2*gamma**2)/(1+gamma**2)))**(1/2)
    a_m = (1-r_m/r_2)**((1-5*gamma**2)/(1+gamma**2))
    phi_m= (1-r_m/r_2)**((4*gamma**2)/(1+gamma**2))
    drho_dr_m = (1+r_m/r_2)**((-1*gamma**2)/(1+gamma**2))-(r_m/r_2)*((-1*gamma**2)/(1+gamma**2))*(1+r_m/r_2)**((-1*gamma**2)/(1+gamma**2)-1)
    b_m = (1-r_m/r_2)**((-1-3*gamma**2)/(1+gamma**2))*(drho_dr_m)**(-2)
    rho_p = ((r_2**2)*(1-(r_m/r_2))**((6*gamma**2)/(1+gamma**2)))**(1/2)
    a_p = (1-r_m/r_2)**((1+3*gamma**2)/(1+gamma**2))
    phi_p= (1-r_m/r_2)**((-4*gamma**2)/(1+gamma**2))
    drho_dr_p = (1+r_m/r_2)**((3*gamma**2)/(1+gamma**2))-(r_m/r_2)*((3*gamma**2)/(1+gamma**2))*(1+r_m/r_2)**((3*gamma**2)/(1+gamma**2)-1)
    b_p = (1-r_m/r_2)**((-1+5*gamma**2)/(1+gamma**2))*(drho_dr_p)**(-2)
    r_lim = 80000
    if option == 1:
        rho_u = rho_m 
        a_u = a_m
        phi_u = phi_m
        drho_dr_u = drho_dr_m
        b_u = b_m
        couleur = (0.85,0.325,0.098)
        nom = 'c --'
    elif option == 2:
        rho_u = rho_p 
        a_u = a_p
        phi_u = phi_p
        drho_dr_u = drho_dr_p
        b_u = b_p
        couleur = (0.929,0.694,0.125)
        nom = 'c -+'
        
    return (r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen)


# In[3]:


def make_plots(option,r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen):
    plt.plot(r*1e-3,a,label=f'a: numerical ({comment})',color=(0.,0.447,0.741))
    plt.plot(rho_u*1e-3,a_u,linestyle='dashed',label = f'a: {nom}',color = couleur)
    plt.axvline(x=radiusStar*1e-3, color='r')
    plt.xlim([0,r_lim*1e-3])
    plt.ylim([a[0],1])
    plt.xlabel('Radius r (km)')
    plt.ylabel('a', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/a_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()
    
    a_interpol = interp1d(rho_u, a_u, fill_value="extrapolate")
    diff_a = (a_interpol(r)-a) / a 


    plt.plot(r*1e-3,diff_a*100,label=f'a (numerical({comment}) - {nom}) in %',color=(0.,0.447,0.741))
    plt.axvline(x=radiusStar*1e-3, color='r')
    plt.axhline(y= 0, color = 'gray')
    plt.xlim([0,r_lim*1e-3])
    # plt.ylim([-0.01,0.01])
    plt.ylim([-1,1])
    plt.xlabel('Radius r (km)')
    plt.ylabel('(a_num - a_ana) %', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/diff_a_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()
    
    plt.plot(r*1e-3,b,label=f'b: numerical ({comment})',color=(0.,0.447,0.741))
    plt.plot(rho_u*1e-3,b_u,linestyle='dashed',label = f'b: {nom}',color = couleur)
    plt.axvline(x=radiusStar*1e-3, color='r')
    plt.xlim([0,r_lim*1e-3])
    plt.ylim([b[0],max(b)+0.1])
    plt.xlabel('Radius r (km)')
    plt.ylabel('b', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/b_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()
    
    b_interpol = interp1d(rho_u, b_u, fill_value="extrapolate")
    diff_b = (b_interpol(r)-b) / b


    plt.plot(r*1e-3,diff_b*100,label=f'b (numerical({comment}) - {nom}) in %',color=(0.,0.447,0.741))
    plt.axvline(x=radiusStar*1e-3, color='r')
    plt.axhline(y= 0, color = 'gray')
    plt.xlim([0,r_lim*1e-3])
    # plt.ylim([-0.01,0.01])
#     plt.ylim([-5,5])
    plt.ylim([-3,3])
    plt.xlabel('Radius r (km)')
    plt.ylabel('(b_num - b_ana) %', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/diff_b_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()
    
    plt.plot(r*1e-3,phi,label=f'$\\Phi$: numerical({comment})',color=(0.,0.447,0.741))
    plt.plot(rho_u*1e-3,phi_u,linestyle='dashed',label= f'$\\Phi$: {nom}',color= couleur)
    plt.axvline(x=radiusStar*1e-3, color='r')
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
    
    phi_interpol = interp1d(rho_u, phi_u, fill_value="extrapolate")
    diff_phi = (phi_interpol(r)-phi) / phi


    plt.plot(r*1e-3,diff_phi*100,label=f'$\\Phi$ (numerical({comment}) - {nom}) in %',color=(0.,0.447,0.741))
    plt.axvline(x=radiusStar*1e-3, color='r')
    plt.axhline(y= 0, color = 'gray')
    plt.xlim([0,r_lim*1e-3])
    # plt.ylim([-0.01,0.01])
    plt.ylim([-1,1])
    plt.xlabel('Radius r (km)')
    plt.ylabel('($\\Phi$_num - $\\Phi$_ana) %', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/diff_phi_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()
    
    plt.plot(r*1e-3,a*b,label=f'a*b: numerical ({comment})',color=(0.,0.447,0.741))
    plt.plot(rho_u*1e-3,a_u*b_u,linestyle='dashed',label = f'a*b: {nom}',color = couleur)
    plt.axvline(x=radiusStar*1e-3, color='r')
    plt.axhline(y= 1, color = 'gray')
    plt.xlim([0,r_lim*1e-3])
    plt.ylim([a[0]*b[0],max(a*b)+0.1])
    plt.xlabel('Radius r (km)')
    plt.ylabel('a * b', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/ab_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()

    ab_interpol = interp1d(rho_u, a_u * b_u, fill_value="extrapolate")
    diff_ab = (ab_interpol(r)- a * b) / (a * b)


    plt.plot(r*1e-3,diff_ab*100,label=f'a*b (numerical({comment}) - {nom}) in %',color=(0.,0.447,0.741))
    plt.axvline(x=radiusStar*1e-3, color='r')
    plt.axhline(y= 0, color = 'gray')
    plt.xlim([0,r_lim*1e-3])
    # plt.ylim([-0.01,0.01])
#     plt.ylim([-1,1])
    plt.ylim([-5,5])
    plt.xlabel('Radius r (km)')
    plt.ylabel('a*b (numerical - analytical) %', fontsize=12)
    plt.title(f'Density = {rho_cen} $MeV/fm^3$, mass (ADM) = {mass_ADM:.1f} solar mass', loc='center')
    plt.legend()
    plt.savefig(f'figures/diff_ab_{descr}_{rho_cen:.0f}_{mass_ADM:.1f}.png', dpi = 200)
    plt.show()


# In[4]:


for i in [1,2]:
    for k in [100,500,1000,2000,4000,8000]:
        r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen  = run(i,k)
        make_plots(i,r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,k)


# In[ ]:




