import numpy as np
from matplotlib import pyplot, cm
import matplotlib.pyplot as plt
from numba import njit, prange
import configparser
from scipy import interpolate
import time
from VAT_ASM_CFD import meshing, solve_pressure_poisson, apply_bc, solve_momentum
from VAT_ASM_BEM import VAT_BEM
from VAT_ASM_Misc import airfoil_data_parser, circular_mask, colorbar


# load config file with user settings
config = configparser.ConfigParser(inline_comment_prefixes='#')
#config.read('config/VAWT850.ini')
# config.read('config/UNHRVAT.ini')
config.read('config/RM2.ini')


def main():
    
    # global mesh settings with linear grow rate and constant-length center area
    n_elements = int(config['mesh settings']['n_elements'])
    radius = float(config['VAWT parameters']['radius']) 

    # limits of the domain
    limits_x = np.array(config['mesh settings']['limits_x'].split(","), dtype=np.float64)
    limits_y = np.array(config['mesh settings']['limits_y'].split(","), dtype=np.float64)

    # linear grow from center area
    grow_rate_x = float(config['mesh settings']['grow_rate_x'])
    grow_rate_y = float(config['mesh settings']['grow_rate_y'])

    # start meshing the domain
    mesh = meshing(n_elements, radius, grow_rate_x, grow_rate_y, limits_x, limits_y)
    x, y, X, Y, nx, ny, dx, dy, grid_area = mesh
    print(str(nx) + " x " + str(ny) + " = " + str(nx * ny) + " mesh nodes")

    # load airfoil data from files
    airfoil = airfoil_data_parser()

    # fluid parameters
    rho = float(config['fluid parameters']['rho'])  # kg/m^3 density
    nu = float(config['fluid parameters']['nu'])  # m^2/s kinematic viscosity

    # simulation parameters
    u_vawt = np.array(config['simulation']['u_vawt'].split(","), dtype=np.float64)
    v0 = np.array(config['simulation']['v0'].split(","), dtype=np.float64)
    Tmax = np.array(config['simulation']['T'].split(","), dtype=np.float64)
    
    try:
        TSR_all = u_vawt / v0
    except:
        raise ValueError("lists for T, TSR and u_vawt need to have the same length")
    print("TSR: " + str(TSR_all))

    # begin simulation
    sim_results = np.zeros((len(TSR_all), 3))
    starttime = time.time()
    for i, TSR in enumerate(TSR_all):
        # instance VAWT class
        VAWT = VAT_BEM(v0[i], u_vawt[i], rho, nu, mesh, airfoil, config)
        sim_results[i, 0] = TSR
        # save results
        sim_results[i, 1], sim_results[i, 2] = VAT_Flow(Tmax[i], rho, nu, v0[i], VAWT, mesh)
        np.savetxt("sim_results.txt", sim_results)
        print("TSR [-]        c_p [-]        Struts [Nm]")
        print(sim_results)
        print("simulation. time: {0:1.6f}s ".format(time.time() - starttime))



def VAT_Flow(Tmax, rho, nu, v0, VAWT, mesh):

    # unpack mesh info
    x, y, X, Y, nx, ny, dx, dy, grid_area = mesh

    # reset simulation data
    dt_f = float(config['simulation']['dt_f'])
    use_tower = int(config['VAWT parameters']['tower'])
    tower_mask = 1 - circular_mask(nx, ny, radius=float(config['VAWT parameters']['tower_width']) / 2)
    closed_sides = int(config['VAWT parameters']['closed_sides'])
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx)) - v0
    p = np.zeros((ny, nx))

    # prepare plot
    fig, (ax, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    fig.canvas.manager.set_window_title('VAT-ASM')
    plt.rc('axes', axisbelow=True)
    fig.set_tight_layout(True)
    plt.show()

    # prepare empty arrays
    un = np.copy(u)
    vn = np.copy(v)
    dudt = np.copy(u)
    dvdt = np.copy(v)
    dudt2 = np.copy(u)
    dvdt2 = np.copy(v)
    plot_data = np.zeros((2000, 3))

    reflow_detected = False

    T, n = 0, 0
    while T <= Tmax:
        # get timings from latest iteration
        start_time = time.time()

        # non-adaptive timestep controls by enforcing CFL condition. Adaptive timestepping is removed currently
        dt = dt_f * (dx / VAWT.u_vawt)

        # Ralston's method
        un = u
        vn = v
        apply_bc(un, vn, v0, nx, ny, use_tower, tower_mask, closed_sides)
        solve_pressure_poisson(p, un, vn, nx, ny, dt, rho, X, Y)
        solve_momentum(dudt, dvdt, un, vn, dx, dy, nx, ny, p, rho, nu, X, Y)
        dudt, dvdt = VAWT.get_acceleration(T, dudt, dvdt, un, vn, 1, dt, mesh)

        # correct
        un = u + 2 / 3 * dt * dudt
        vn = v + 2 / 3 * dt * dvdt
        apply_bc(un, vn, v0, nx, ny, use_tower, tower_mask, closed_sides)
        solve_pressure_poisson(p, un, vn, nx, ny, 2 / 3 * dt, rho, X, Y)
        solve_momentum(dudt2, dvdt2, un, vn, dx, dy, nx, ny, p, rho, nu, X, Y)
        dudt2, dvdt2 = VAWT.get_acceleration(T + 2 / 3 * dt, dudt2,dvdt2, un, vn, 2, dt, mesh)

        # merge
        u = u + dt * (1 / 4 * dudt + 3 / 4 * dudt2)
        v = v + dt * (1 / 4 * dvdt + 3 / 4 * dvdt2)
        
        # detect reflow into the domain from the bottom. Build wall if necessary
        if np.max(v[0, :]) > 0:
            v[0, :] = np.where(v[0, :] > 0, v[0, :] * 0, v[0, :])
            reflow_detected = True

        T = T + dt

        # update plot every X timesteps. Plotting slows down performance
        # considerably
        if np.mod(n, int(config['plot settings']['n_plot'])) == 0 or T >= Tmax:

            # show mean values
            print("--- step " + str(n) + " finished ---")
            print("calc. time: {0:1.6f}s ".format(time.time() - start_time))
            print("sim. time: {0:3.2f}s".format(T))
            print("timestep: {0:1.6f}s".format(dt))
            P = np.mean(np.sum(VAWT.P, axis=1))
            print("power: {0:3.2f}W".format(P))
            plot_data = np.roll(plot_data, 1, axis=0)
            plot_data[0, 0] = n
            plot_data[0, 1] = np.mean(np.sum(VAWT.eta, axis=1)) * 100
            plot_data[0, 2] = np.mean(np.sum(VAWT.torque_struts, axis=1))
            print("cp: {0:2.2f}%".format(plot_data[0, 1]))
            print("struts : {0:2.2f}Nm".format(plot_data[0, 2]))

            # project solution for aquidistant grid for plotting
            fu = interpolate.interp2d(x, y, u, kind='linear')
            fv = interpolate.interp2d(x, y, v, kind='linear')
            # fp = interpolate.interp2d(x, y, p, kind='linear') # pressure


            xnew = np.arange(np.min(x), np.max(x), dx)
            ynew = np.arange(np.min(y), np.max(y), dy)
            unew = fu(xnew, ynew)
            vnew = fv(xnew, ynew)
            # pnew = fp(xnew,ynew)
   

            plt.subplot(131)
            im = plt.pcolormesh(xnew,ynew,np.sqrt(unew**2 +vnew**2),alpha=1,vmin=0,
                vmax=v0 * 1.6, cmap=cm.inferno, shading='auto')
            # im = plt.pcolormesh(xnew, ynew, pnew , alpha=1, cmap=cm.inferno,vmin=-900,vmax=900) # show pressure
            # im = plt.pcolormesh(xnew, ynew, Wnew , alpha=1, cmap=cm.inferno,vmin=0,vmax=0.1) # show weights
            cbar = colorbar(im)
            cbar.ax.set_title('$v$[m/s]')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')

            # user plot limits
            limits_x = np.array(config['plot settings']['limits_x'].split(","), dtype=np.float64)
            limits_y = np.array(config['plot settings']['limits_y'].split(","), dtype=np.float64)
            plt.xlim((limits_x[0], limits_x[1]))
            plt.ylim((limits_y[0], limits_y[1]))

            # mesh lines
            show_grid = int(config['plot settings']['show_grid'])
            if show_grid == True:
                plt.plot(X, Y, c="black", linewidth=0.2)
                plt.plot(X.T, Y.T, c="black", linewidth=0.2)

            # streamlines or velocity arrows. Very slow!
            # plt.quiver(X[30:-30,30:-30], Y[30:-30,30:-30], u[30:-30,30:-30], v[30:-30,30:-30],color = "white")
            # plt.streamplot(xnew, ynew, unew, vnew,color="white",density = 3)

            # lift and drag plot
            plt.subplot(132)
            plt.grid()
            for blade in range(0, VAWT.z_blades):
                plt.scatter(np.rad2deg(VAWT.alpha_A[:, blade]), VAWT.cl[:, blade])
                # plt.scatter(np.rad2deg(VAWT.alpha_A[:,blade]),VAWT.cd[:,blade])
            # plt.xlim((np.min(np.rad2deg(VAWT.alpha_A[:,0])),np.max(np.rad2deg(VAWT.alpha_A[:,0]))))
            plt.ylim((-2.2, 2.2))
            plt.ylabel('Lift Coefficient, $c_l$')
            plt.xlabel('Angle of Attack, α [deg]')

            # # show wake velocity profile behind turbine
            # plt.subplot(132)
            # y_wake = np.where(y <= -50)[0][-1] # not exact
            # plt.plot(x,-v[y_wake,:])
            # plt.ylabel('Velocity, $v_y [m/s]$')
            # plt.xlabel('X [m]')

            # eta over time
            plt.subplot(133)
            plt.grid()
            plt.scatter(plot_data[:, 0], plot_data[:, 1], c="black")
            plt.xlabel('Iterations')
            plt.ylabel('Turbine Efficiency, $c_p$')

            # plot torque over phi
            # plt.subplot(133)
            # plt.scatter(np.rad2deg(VAWT.phi[:,0]),VAWT.Torque[:,0],c="red", vmin=-2, vmax=2)
            # plt.scatter(np.rad2deg(VAWT.phi[:,1]),VAWT.Torque[:,1],c="blue", vmin=-2, vmax=2)

            if reflow_detected:
                print("reflow detected: placing walls")
                reflow_detected = False

            if T < Tmax:
                fig.canvas.draw()
                fig.canvas.flush_events()
                # pyplot.cla()
                pyplot.clf()

        n = n + 1

    return plot_data[0, 1], plot_data[0, 2]


if __name__ == '__main__':
    main()
