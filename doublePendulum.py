# -*- coding: utf-8 -*-
"""

Author: JS
Created: Sat Nov  1 12:13:46 2025
Modified: Sat Nov  1 12:13:46 2025

Description
-----------

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numba import njit, prange

## Parameters

m1 = 1
m2 = 1
l1 = 1
l2 = 1
theta1_0 = np.radians(150)
theta2_0 = np.radians(-80)
g = 9.81

w1_0 = 0
w2_0 = 0

t0 = 0
tN = 30
h = 0.01

N = int((tN - t0) // h)

t = np.linspace(t0, tN, N + 1)

## Double Pendulum Equations

# def w1dot(y):
    
#     theta1 = y[0]
#     w1 = y[1]
#     theta2 = y[2]
#     w2 = y[3]
    
#     delta = theta2 - theta1
    
#     return((m2*l1*np.sin(delta)*np.cos(delta)*w1**2 + m2*g*np.sin(theta2)*np.cos(delta) + 
#             m2*l2*np.sin(delta)*w2**2 - (m1 + m2)*g*np.sin(theta1)) / 
#            ((m1 + m2)*l1 - m2*l1*(np.cos(delta))**2))

# def w2dot(y):
    
#     theta1 = y[0]
#     w1 = y[1]
#     theta2 = y[2]
#     w2 = y[3]
    
#     delta = theta2 - theta1
    
#     return((-m2*l2*np.sin(delta)*np.cos(delta)*w2**2 + (m1 + m2)*(g*np.sin(theta1)*np.cos(delta) - 
#             l1*np.sin(delta)*w1**2 - g*np.sin(theta2))) / 
#            ((m1 + m2)*l1 - m2*l1*(np.cos(delta))**2))

# def theta1dot(y): return(y[1])

# def theta2dot(y): return(y[3])

## Define our state vector

y0 = np.array([theta1_0, w1_0, theta2_0, w2_0], dtype=float)

@njit
def ydot(y):
    
    theta1 = y[0]
    w1 = y[1]
    theta2 = y[2]
    w2 = y[3]
    
    delta = theta2 - theta1
    
    return np.array([y[1],
                    (m2*l1*np.sin(delta)*np.cos(delta)*w1**2 + m2*g*np.sin(theta2)*np.cos(delta) +
                    m2*l2*np.sin(delta)*w2**2 - (m1 + m2)*g*np.sin(theta1)) / 
                    ((m1 + m2)*l1 - m2*l1*(np.cos(delta))**2),
                    y[3], 
                    (-m2*l2*np.sin(delta)*np.cos(delta)*w2**2 + (m1 + m2)*(g*np.sin(theta1)*np.cos(delta) - 
                    l1*np.sin(delta)*w1**2 - g*np.sin(theta2))) / 
                    ((m1 + m2)*l1 - m2*l1*(np.cos(delta))**2)])

## RK4 method for four coupled ODEs

def RK4_DP(y0, t, h):
       
    y = np.zeros((len(t), 4))
    
    y[0] = y0

    for i in range(0, len(t) - 1):

        y_k1 = ydot(y[i])
        y_k2 = ydot(y[i] + h*y_k1/2)
        y_k3 = ydot(y[i] + h*y_k2/2)
        y_k4 = ydot(y[i] + h*y_k3)

        y[i+1] = y[i] + h * (y_k1 + 2 * (y_k2 + y_k3) + y_k4) / 6

    return(y)

# print(RK4_DP(y0, t, h))

## Plot of the trajectory of the Double Pendulum

# trajectory = RK4_DP(y0, t, h)

# fig = plt.figure(0, figsize = (8,6))
# ax0 = fig.add_subplot(1,1,1)

# X1 = l1 * np.sin(trajectory[:, 0])
# Y1 = - l1 * np.cos(trajectory[:, 0])

# X2 = X1 + l2 * np.sin(trajectory[:, 2])
# Y2 = Y1 - l2 * np.cos(trajectory[:, 2])

# # ax0.plot(X1, Y1, color='blue')
# ax0.plot(X2, Y2, color='red')


# ax0.set_xlabel('X')
# ax0.set_ylabel('Y')
# ax0.set_aspect('equal')

# plt.show()

## Double Pendulum animation

# from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_xlim(-(l1+l2+0.5), l1+l2+0.5)
# ax.set_ylim(-(l1+l2+0.5), l1+l2+0.5)
# ax.set_aspect('equal')

# line, = ax.plot([], [], 'o-', lw=2)  # line connecting the bobs
# trace, = ax.plot([], [], '-', lw=1, color='lightgray')  # optional path trace

# # Store previous positions for trace
# trace_x, trace_y = [], []

# def init():
#     line.set_data([], [])
#     trace.set_data([], [])
#     return line, trace

# def update(i):
#     x = [0, X1[i], X2[i]]
#     y = [0, Y1[i], Y2[i]]
#     line.set_data(x, y)
    
#     # Append for trace
#     trace_x.append(X2[i])
#     trace_y.append(Y2[i])
#     trace.set_data(trace_x, trace_y)
    
#     return line, trace

# ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=10)
# plt.show()

## Defining how 'chaotic' the pendulum becomes

@njit
def RK4_DP_F(y0, t, h):
       
    y = np.zeros((len(t), 4))
    
    y[0] = y0

    for i in range(0, len(t) - 1):

        y_k1 = ydot(y[i])
        y_k2 = ydot(y[i] + h*y_k1/2)
        y_k3 = ydot(y[i] + h*y_k2/2)
        y_k4 = ydot(y[i] + h*y_k3)

        y[i+1] = y[i] + h * (y_k1 + 2 * (y_k2 + y_k3) + y_k4) / 6

    return(y[len(t) - 1])

@njit(parallel=True)
def chaos_angle(w1_var, w2_var, t, h, num):
    
    theta1_var = np.linspace(-np.pi, np.pi, num)
    theta2_var = np.linspace(-np.pi, np.pi, num)

    
    chaos_index = np.zeros((num, num))
    
    y_var = np.empty(4)
    y_var_chaos = np.empty(4)
    
    for i in prange(num):
        for j in range(num):
            
            y_var = np.array([theta1_var[i], w1_var, theta2_var[j], w2_var])
            
            y_var_chaos = np.array([theta1_var[i] + 1e-8, w1_var, theta2_var[j] + 1e-8, w2_var])
            
            y_fin = RK4_DP_F(y_var, t, h)
            
            y_fin_chaos = RK4_DP_F(y_var_chaos, t, h)
            
            chaos_index[i][j] = np.sqrt(np.sum((y_fin - y_fin_chaos)**2))
            
    return(chaos_index)

@njit(parallel=True)
def chaos_freq(theta1_var, theta2_var, t, h, num):
    
    w1_var = np.linspace(-40, 40, num)
    w2_var = np.linspace(-40, 40, num)

    
    chaos_index = np.zeros((num, num))
    
    y_var = np.empty(4)
    y_var_chaos = np.empty(4)
    
    for i in prange(num):
        for j in range(num):
            
            y_var = np.array([theta1_var, w1_var[i], theta2_var, w2_var[i]])
            
            y_var_chaos = np.array([theta1_var, w1_var[i] + 1e-8, theta2_var, w2_var[j] + 1e-8])
            
            y_fin = RK4_DP_F(y_var, t, h)
            
            y_fin_chaos = RK4_DP_F(y_var_chaos, t, h)
            
            chaos_index[i][j] = np.sqrt(np.sum((y_fin - y_fin_chaos)**2))
            
    return(chaos_index)

## Plot of the 'chaos' heat map

theta1 = 0
theta2 = 0
w1 = 0
w2 = 0

chaos_index_angle = chaos_angle(w1, w2, t, h, 1000)
chaos_index_freq = chaos_freq(theta1, theta2, t, h, 1000)

fig1 = plt.figure(1, figsize=(6,5))
ax1 = fig1.add_subplot(1,1,1)

dark_blue = "#030021"
white = "#FFFFFF"

cmap = mcolors.LinearSegmentedColormap.from_list("darkblue_white", [dark_blue, white])

im1 = ax1.imshow(np.log10(chaos_index_angle + 1e-8),
                extent=[-180,180,-180,180],
                origin='lower', cmap=cmap, aspect='auto')

cbar = fig1.colorbar(im1, ax=ax1)
cbar.set_label(r'$\log_{10}(\Delta)$')
ax1.set_xlabel(r'$\theta_1$ (deg)')
ax1.set_ylabel(r'$\theta_2$ (deg)')

filename = f"DP_angle_{w1}_{w2}.png"
plt.savefig(filename, dpi=600)

plt.show()


fig2 = plt.figure(2, figsize=(6,5))
ax2 = fig2.add_subplot(1,1,1)

cmap = mcolors.LinearSegmentedColormap.from_list("darkblue_white", [dark_blue, white])

im2 = ax2.imshow(np.log10(chaos_index_freq + 1e-8),
                extent=[-180,180,-180,180],
                origin='lower', cmap=cmap, aspect='auto')

cbar = fig2.colorbar(im2, ax=ax2)
cbar.set_label(r'$\log_{10}(\Delta)$')
ax2.set_xlabel(r'$\omega_1$ (Hz)')
ax2.set_ylabel(r'$\omega_2$ (Hz)')

filename = f"DP_freq_{w1:.2f}_{w2:.2f}.png"
plt.savefig(filename, dpi=600)

plt.show()


        