# Import library yang dibutuhkan
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import numpy as np

# Définir la voiture à pendule comme une classe
# L : longueur de la corde
# mP : masse du pendule
# mC : masse de la voiture
# dP : vitesse initiale du pendule
# dC : vitesse initiale de la voiture
# g : gravité
class PendulumCart():
    def __init__(self, L=1.0, mP=0.0, mC=10.0, dP=1.0, dC=5.0, g=9.81, action_range=10):
        self.L = L
        self.mP = mP
        self.mC = mC
        self.dP = dP
        self.dC = dC
        self.g = g
        self.action_range = action_range

    def _ode(self, X0, t=0,  inputT=0, inputF=0 , interp_F=True):
        x = X0[0]
        dx = X0[1]
        theta = X0[2]
        dtheta = X0[3]

        if interp_F is list:
            F = np.interp(t, inputT, inputF)
        else: F = inputF
        
        s = np.sin(theta)
        c = np.cos(theta)

        # num1 = F - (self.dC * dx) + (self.mP * self.L * s * dtheta ** 2.0)
        # num1 = F + (self.mP * self.L * s * dtheta ** 2.0)
        # num2 = c * self.g * s + -(c * self.dP * dtheta) / (self.L)
        # num3 = (self.mC + self.mP) * self.g * s + -(self.mC + self.mP) * (self.dP * dtheta) / (self.mP * self.L)
        # den = self.mC - (self.mP * s**2.0)

        # ddx = (num1 + num2) / den
        # ddtheta = (1/self.L) * (num1 * c + num3) / den

        den = self.mC + (self.mP * s**2.0)

        ddx = (F - self.mP * self.L * dtheta**2 * s + self.mC * self.g * s * c) / den
        ddtheta = (self.g * s + ddx * c) / self.L

        dX = np.zeros(np.size(X0))
        dX[0] = dx
        dX[1] = ddx
        dX[2] = dtheta
        dX[3] = ddtheta

        return dX

    def simulate(self, x_init, t, inputT, inputF):
        dt = t[1] - t[0]
        n_steps = len(t)
        X = np.zeros((n_steps, len(x_init)))
        X[0, :] = x_init

        for i in range(1, n_steps):
            t_curr = t[i - 1]
            X_curr = X[i - 1, :]

            # Étapes intermédiaires (RK4)
            k1 = self._ode(X_curr, t_curr, inputT, inputF)
            k2 = self._ode(X_curr + dt * k1 / 2, t_curr + dt / 2, inputT, inputF)
            k3 = self._ode(X_curr + dt * k2 / 2, t_curr + dt / 2, inputT, inputF)
            k4 = self._ode(X_curr + dt * k3, t_curr + dt, inputT, inputF)

            # Mise à jour de l'état
            X[i, :] = X_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return X
    def next_state(self, X_curr, dt, idx_action):
        F = [-5, 5][idx_action]
        # Étapes intermédiaires (RK4)
        k1 = self._ode(X_curr, inputF=F)
        k2 = self._ode(X_curr + dt * k1 / 2, inputF=F)
        k3 = self._ode(X_curr + dt * k2 / 2, inputF=F)
        k4 = self._ode(X_curr + dt * k3, inputF=F)

        # Mise à jour de l'état
        next_X = X_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Wrap angle to stay between -pi and pi
        next_X[2] = ((next_X[2] + np.pi) % (2 * np.pi)) - np.pi
        return next_X
        

class DrawCart:
    def __init__(self, ax, cart):
        with plt.style.context("dark_background"):
            self.body1, = ax.plot([], [])
            self.body2, = ax.plot([ ], [ ])
            self.arm, = ax.plot([], [])
            self.pen, = ax.plot([], [])
            self.ax = ax
            self.cart = cart
            ax.grid(True, linestyle="-", alpha=0.05, color = 'black')

    def draw(self, pos, theta):
        # bh, bv = 0.1, 2.2
        # bodyx = np.array([ -1.0, -1.0, 1.0, 1.0, -1.0 ]) * bh + pos
        # bodyy = np.array([ -0.9, 1.1, 1.1, -0.9, -0.9 ]) * bv
        # self.body1.set_data(bodyx, bodyy)

        bh, bv = 0.2, 0.2
        bodyx = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])*bh + pos
        bodyy = np.array([-1, 1, 1, -1, -1])*bv
        self.body2.set_data(bodyx, bodyy)

        xp, yp = pos - self.cart.L * np.sin(theta), self.cart.L * np.cos(theta)
        self.arm.set_data([pos, xp], [0, yp])

        phi = np.linspace(-np.pi, np.pi, 32)
        d = self.cart.mP/self.cart.mC

        self.pen.set_data(xp + d*np.cos(phi), yp + d*np.sin(phi))

        self.ax.set_xlim(- 8, + 8)
        self.ax.set_ylim(-4, 4)
        self.ax.set_aspect('equal')
        return self.ax, self.body1, self.body2, self.arm, self.pen

if __name__ == '__main__':
# Example of RBL case
    cart = PendulumCart(L=3, mP=2, mC=10, dP=1, dC=5)

    T0 = 0.0
    TN = 20.0
    t = np.linspace(T0, TN, num=5000)

    x_init = [0.0, 0.0, 0.6, 0.0]

    inputT = [T0, 5.0, 5.001, 5.5, 5.501, TN]
    inputF = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    x = cart.simulate(x_init, t, inputT, inputF)

    with plt.style.context("ggplot"):
        color1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        color2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
        color3 = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
        color4 = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
        color5 = plt.rcParams['axes.prop_cycle'].by_key()['color'][4]
        color6 = plt.rcParams['axes.prop_cycle'].by_key()['color'][5]

        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(2, 2, figure=fig,height_ratios=[1.5, 1],width_ratios=[2, 2], wspace = 0.5)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax11 = ax1.twinx()
        ax11.grid(None)
        ax2 = fig.add_subplot(gs[1, 1])
        ax21 = ax2.twinx()
        ax21.grid(None)
        #ax3 = fig.add_subplot(gs[1, 2])

    points = np.full(5, None)

    ax1.plot(t, x[:, 0], color=color1)

    ax1.set_ylabel(r'x (m)', color=color1)
    ax11.plot(t, x[:, 1], color=color2)

    ax11.set_ylabel(r'v ($\frac{m}{s}$)', color=color2)

    ax2.plot(t, x[:, 2], color=color3)
    ax2.set_ylabel(r'$\theta$ (rad)', color=color3)
    ax21.plot(t, x[:, 3], color=color4)
    ax21.set_ylabel(r'$\omega$ ($\frac{rad}{s}$)', color=color4)

    #ax3.plot(inputT, inputF, ':', color=color5)
    #ax3.set_ylabel(r'F (N)', color=color5)

    dc = DrawCart(ax0, cart)

    points[0], = ax1.plot([], [], 'o', color=color1)
    points[1], = ax11.plot([], [], 'o', color=color2)
    points[2], = ax2.plot([], [], 'o', color=color3)
    points[3], = ax21.plot([], [], 'o', color=color4)
    #points[4], = ax3.plot([], [], 'o', color=color5)


    def animate(i):
        time = (i % 400) * 0.1
        pos = np.interp(time, t, x[:, 0])
        dpos = np.interp(time, t, x[:, 1])
        theta = np.interp(time, t, x[:, 2])
        dtheta = np.interp(time, t, x[:, 3])
        force = np.interp(time, inputT, inputF)

        points[0].set_data([time], [pos])
        points[1].set_data([time], [dpos])
        points[2].set_data([time], [theta])
        points[3].set_data([time], [dtheta])
        #points[4].set_data(time, force)

        ax, l1, l2, l3, l4 = dc.draw(pos, theta)

        return ax, points[0], points[1], points[2], points[3], #points[4],


    ani = FuncAnimation(fig, animate, interval=20, save_count=200)
    #ani.save('odePendulumCart.gif', writer='imagemagick')

    plt.show()