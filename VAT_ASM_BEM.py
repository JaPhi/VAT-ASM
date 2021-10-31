import numpy as np
from numba import njit, prange
from VAT_ASM_Misc import bilinear_interp, airfoil_data



class VAT_BEM:
    def __init__(self, v0, u_vawt, rho, nu, mesh, airfoil, config):

        # mesh and airfoil data unpacked
        self.x, self.y, self.X, self.Y, self.nx, self.ny, self.dx, self.dy, self.grid_area = mesh
        self.airfoil = airfoil

        # base VAWT user parameters
        self.radius = float(config['VAWT parameters']['radius'])
        self.height = float(config['VAWT parameters']['height'])
        self.chord = float(config['VAWT parameters']['chord'])
        self.z_blades = int(config['VAWT parameters']['z_blades'])
        self.z_struts = int(config['VAWT parameters']['z_struts'])
        self.struts_width = float(config['VAWT parameters']['struts_width'])
        self.center_pos = np.array(config['VAWT parameters']['center_pos'].split(","), dtype=np.float64)
        self.h = self.dx * float(config['VAWT parameters']['h'])

        # general data
        self.u_vawt = u_vawt
        self.nu = nu
        self.rho = rho
        self.omega = self.u_vawt / self.radius
        self.P_max = 0.5 * rho * 2 * (self.radius + self.h / 2) * self.height * v0**3  # - self.chord * 0.21 * 0.5

        # higher order corrections
        self.dynamic_stall = int(config['higher order corrections']['dynamic_stall'])
        self.flow_curvature = int(config['higher order corrections']['flow_curvature'])
        self.aspect_ratio = int(config['higher order corrections']['aspect_ratio'])
        self.parasitic_torque = int(config['higher order corrections']['parasitic_torque'])

        # save n simulation parameters for "ave_rot" rotations. Problematic with adaptive timestep.
        ave_rot = float(config['simulation']['ave_rot'])
        dt_f = float(config['simulation']['dt_f'])
        n_saves = int(ave_rot * ((2 * np.pi) / (self.u_vawt / self.radius)) / (dt_f * self.dx / self.u_vawt))
        self.phi = np.zeros((n_saves, self.z_blades))
        self.P = np.zeros((n_saves, self.z_blades))
        self.Torque = np.zeros((n_saves, self.z_blades))
        self.cl = np.zeros((n_saves, self.z_blades))
        self.cd = np.zeros((n_saves, self.z_blades))
        self.eta = np.zeros((n_saves, self.z_blades))
        self.alpha_A = np.zeros((n_saves, self.z_blades))
        self.fd = np.zeros((2, self.z_blades))
        self.torque_struts = np.zeros((n_saves, self.z_blades))

        # simulation data
        self.acc_x = np.zeros((self.ny, self.nx))
        self.acc_y = np.zeros((self.ny, self.nx))
        self.W = np.zeros((self.ny, self.nx))

        # precalculate struts bounding box indices within the machine diameter
        self.struts_where_x = np.array([np.where(
            self.x < -self.radius + self.center_pos[0])[0][-1], np.where(self.x > self.radius + self.center_pos[0])[0][0]])
        self.struts_where_y = np.array([np.where(
            self.y < -self.radius + self.center_pos[1])[0][-1], np.where(self.y > self.radius + self.center_pos[1])[0][0]])


    def get_parasitic_torque(self,pos_x,pos_y,phi,u,v,blade,struts_width):
        """ calculates to torque from the struts in a post processing step. It has
            no direct influence on the velocity field. The struts use the same 
            blade profile as the blades here. 

        Args:
            pos_x, pos_y: absolute blade coordinates
            phi: circumferential blade angle 0...360°
            u, v: velocity components
            blade: current blade number
            struts_width: struts chord length
                   
        Returns:
            torque_struts: sum of all struts
        """

        # uses 10 points along the strut
        struts = np.zeros((10, 3))
        struts[:, 0] = np.linspace(self.center_pos[0], pos_x, 10)  # x-pos
        struts[:, 1] = np.linspace(self.center_pos[1], pos_y, 10)  # y-pos
        struts[:, 2] = struts_width

        r = np.linspace(0, self.radius, 10)
        dr = self.radius / (10 - 1)

        u_struts = r * self.omega

        u2_x = u_struts * np.sin(phi)
        u2_y = -u_struts * np.cos(phi)

        # bounding box for the velocity interpolation based on the rough struts
        # position. Saves performance over evaluating the whole domain.
        s1x = self.struts_where_x[0] - 1
        s2x = self.struts_where_x[1] + 1
        s1y = self.struts_where_y[0] - 1
        s2y = self.struts_where_y[1] + 1

        # bilinear interpolation for each strut segment
        v2_x = bilinear_interp(struts[:, 0], struts[:, 1], self.x[s1x:s2x], self.y[s1y:s2y], u[s1y:s2y, s1x:s2x])
        v2_y = -bilinear_interp(struts[:, 0], struts[:, 1], self.x[s1x:s2x], self.y[s1y:s2y], v[s1y:s2y, s1x:s2x])

        c2_x = u2_x + v2_x
        c2_y = u2_y - v2_y
        c2 = np.sqrt(c2_x**2 + c2_y**2)

        Re = (c2 * struts[:, 2]) / self.nu

        alpha_u = np.abs(np.arccos((c2_x * u2_x + c2_y * u2_y) /
                                   (np.sqrt(c2_x**2 + c2_y**2) * np.sqrt(u2_x**2 + u2_y**2))))
        alpha_u = np.nan_to_num(alpha_u)

        c2_perp = c2 * np.cos(alpha_u)

        cd_list = np.copy(r)
        for i in range(0, len(cd_list)):
            cd_list[i] = airfoil_data(0, Re[i], self.airfoil, return_cd=True)  # alpha = 0 for struts

        struts_torque = 0.5 * self.rho * \
            c2_perp**2 * struts[:, 2] * dr * cd_list * r

        struts_torque = struts_torque * self.z_struts
        self.torque_struts[0, blade] = np.sum(struts_torque)

    

    @staticmethod
    @njit(nopython=True, parallel=False)
    def get_weights(pos_x, pos_y, index_x, index_y, mesh, phi, c, h):
        """ returns the normalized weight matrix to distribute the calculated blade
            forces into the domain. W has the same shape as u and v.

        Args:
            pos_x, pos_y: absolute blade coordinates
            index_x, index_y: index of the next mesh element
            mesh: packed mesh info
            phi: circumferential blade angle 0...360°
            c: chord length
            h: smoothing lenght user parameter
         
        Returns:
            pos_x, pos_y: absolute blade coordinates
            index_x, index_y: index of the next mesh element
        """

        x, y, X, Y, nx, ny, dx, dy, grid_area = mesh

        W = np.zeros((ny, nx))

        # limits the area around the blade to static 3*c to speed up
        # calculation when using a kernel function
        B = np.zeros((int(3 * c / dy), int(3 * c / dx)))

        X_box = X[index_y-int(B.shape[0]/2):index_y+int(B.shape[0]/2)+1,
                  index_x-int(B.shape[1]/2):index_x+int(B.shape[1]/2)+1]
        Y_box = Y[index_y-int(B.shape[0]/2):index_y+int(B.shape[0]/2)+1,
                  index_x-int(B.shape[1]/2):index_x+int(B.shape[1]/2)+1] 

        angle = phi - np.pi

        weights = np.zeros(np.shape(X_box))

        # The first index describes the normalized blade position from the
        # strut connection point. The second index is the weight associated to
        # that point.
        p_dist = np.array([[-0.5, 2], [-0.375, 1.5], [-0.25, 1], [-0.125, 0.833], [0, 0.666], [
                          0.125, 0.5], [0.25, 0.33], [0.375, 0.1666], [0.5, 0]])  # strut attached to the blade center, normalized [-0.5 -> 0.5]
        for i in range(0, 9):

            posX = (c * p_dist[i, 0]) * np.sin(-angle) + pos_x
            posY = (c * p_dist[i, 0]) * np.cos(-angle) + pos_y

            r_x = (X_box - posX)
            r_y = (Y_box - posY)

            r = np.sqrt(r_x**2 + r_y**2)

            # normalized epechenikov kernel
            pre_poly6 = 3 / 4 * (1 - (r / h)**2)  
            pre_poly6 = pre_poly6 * np.where(r <= h, 1, 0) * p_dist[i, 1]
            weights = weights + pre_poly6

        weights = weights / (np.sum(weights))

        W[index_y - int(B.shape[0] / 2):index_y + int(B.shape[0] / 2) + 1,
          index_x - int(B.shape[1] / 2):index_x + int(B.shape[1] / 2) + 1] = weights

        return W

    # return the x and y position of the blade and the corresponding indices
    # on the grid
    def get_position(self, blade, T):
        """ calculate blade position and next index in the mesh

        Args:
            blade: current blade number
            T: absolute time
         
        Returns:
            pos_x, pos_y: absolute blade coordinates
            index_x, index_y: index of the next mesh element
        """

        # pseudo time for all blades after the first
        addT = 2 * np.pi * self.radius / self.u_vawt / self.z_blades
        self.phi[0, blade] = np.mod(
            (T + blade * addT) * self.u_vawt / self.radius, 2 * np.pi)

        # virtual blade offset to compensate for smoothing-lenght-leakage over
        # the machine diameter
        r = self.radius
        phi = self.phi[0, blade]

        pos_x = (r * np.cos(phi))
        pos_y = -(r * np.sin(phi))

        pos_x = pos_x + self.center_pos[0]
        pos_y = self.center_pos[1] - pos_y

        index_x = np.where(self.x > pos_x)[0][0]
        index_y = np.where(self.y > pos_y)[0][0]

        return pos_x, pos_y, index_x, index_y


    def BEM(self, blade, u, v, W, dt):
        """ classic blade element theory with interpolated cl and cd values from tabular data
            improved by higher order corrections

        Args:
            blade: current blade number
            u, v: velocity components
            W: weight matrix for current blade
           
        Returns:
            Fx, Fy: blade forces 
        """

        u_vawt = self.u_vawt
        r = self.radius
        c = self.chord

        phi = self.phi[0, blade]
        rho = self.rho

        v2_x = np.sum(u * W)
        v2_y = np.sum(v * W)

        pos_x = r * np.cos(phi)
        pos_y = r * np.sin(phi)

        u2_x = u_vawt * np.sin(phi)
        u2_y = -u_vawt * np.cos(phi)

        c2_x = u2_x + v2_x
        c2_y = u2_y + v2_y

        c2 = np.sqrt(c2_x**2 + c2_y**2)

        Re = (c2 * self.chord) / self.nu

        alpha_sign = np.arccos((c2_x * pos_x + c2_y * pos_y) /
                               (np.sqrt(c2_x**2 + c2_y**2) * np.sqrt(pos_x**2 + pos_y**2)))
        alpha_sign = 1 if alpha_sign > np.pi / 2 else -1
        alpha_u = np.abs(np.arccos((c2_x * u2_x + c2_y * u2_y) /
                                   (np.sqrt(c2_x**2 + c2_y**2) * np.sqrt(u2_x**2 + u2_y**2))))
        alpha_u = alpha_u * alpha_sign

        # Flow curvature correction Goude
        alpha_goude = 0
        if self.flow_curvature == True:
            alpha_goude = (self.omega * self.chord) / (4 * self.omega * self.radius)

        self.alpha_A[0, blade] = alpha_u  # - pitch if present

        # AR-Correction Prantl (1/2)
        if self.aspect_ratio == True:
            cl = airfoil_data(np.rad2deg(self.alpha_A[0, blade] + alpha_goude),
                              Re, self.airfoil, return_cl=True)
            AR = self.height / self.chord
            e = cl / (np.pi * AR)
            self.alpha_A[0, blade] = self.alpha_A[0, blade] - e 

        cl, cd = airfoil_data(np.rad2deg(self.alpha_A[0, blade] + alpha_goude),
                              Re, self.airfoil, return_cl=True, return_cd=True)

        # Dynamic Stall Oye
        if self.dynamic_stall == True:
            # for symmetric profiles only the flow curvature changes the zero lift angle a0
            a0 = -alpha_goude
            cla0 = airfoil_data(np.rad2deg(alpha_goude), Re, self.airfoil, return_cl=True)

            slope_Clfa = airfoil_data(0, Re, self.airfoil, return_slope=True)
            Clfa = slope_Clfa * np.rad2deg(self.alpha_A[0, blade]) + cla0

            asep = 32 * np.sign(cl)  # 32° from Literature
            Clasep = airfoil_data(asep, Re, self.airfoil, return_cl=True)

            t0 = (self.alpha_A[0, blade] - a0) / (asep - a0)
            t1 = (self.alpha_A[0, blade] - asep) / (asep - a0)

            # hermite spline is only defined up to asep. uses static lift polar after that angle
            if np.abs(self.alpha_A[0, blade]) < np.abs(asep):
                Clfs = t0 * ((asep - a0) * 0.5 * slope_Clfa * (1 + t0 *
                                                               ((7 / 6) * t1 - 1)) + Clasep * t0 * (1 - 2 * t1))
            else:
                Clfs = cl

            f = (cl - Clfs) / (Clfa - Clfs)
            tau = 8 * self.chord / (2 * c2)  # default 8

            fd = (self.fd[1, blade] * (tau - dt) + f * dt) / tau
            fd = np.nan_to_num(fd)

            cl = fd * Clfa + (1 - fd) * Clfs

            self.fd[0, blade] = fd

        # AR-Correction Prantl (2/2)
        if self.aspect_ratio == True:
            cd = cd + cl**2 / (np.pi * AR)

        c_n = cl * np.cos(alpha_u) + cd * np.sin(alpha_u)
        c_t = cl * np.sin(alpha_u) - cd * np.cos(alpha_u)

        Fy = 0.5 * rho * c2**2 * c * 1.0 * (c_n * np.sin(phi) - c_t * np.cos(phi))
        Fx = 0.5 * rho * c2**2 * c * 1.0 * (c_n * np.cos(phi) + c_t * np.sin(phi))

        torque = 0.5 * rho * c2**2 * c * self.height * c_t * r
        P = (torque - self.torque_struts[0, blade]) * self.omega

        # save current values
        self.P[0, blade] = P
        self.cl[0, blade] = cl
        self.cd[0, blade] = cd
        self.eta[0, blade] = P / self.P_max
        self.Torque[0, blade] = P / (u_vawt / r)

        return Fx, Fy


    def get_acceleration(self, T, dudt, dvdt, u, v, k, dt, mesh):
        """ main part of the VAT class to access all methods to get the
            accelerations for the global velocity field

        Args:
            T: absolute simulation time
            dudt, dvdt: incoming current fluid acceleration due to fluid forces
            u, v: velocity components
            k: current Runge-Kutta step
            mesh: packed mesh info
            
        Returns:
            dudt, dvdt: projects the blade forces as an acceleration term into the 
                        fluid solvers data
        """

        u_vawt = self.u_vawt
        r = self.radius
        z = self.z_blades
        rho = self.rho
        if k == 1:
            # Only update data for the first K in RK2, which equals a new true timestep.
            # Improves performance
            self.phi = np.roll(self.phi, 1, axis=0)
            self.Torque = np.roll(self.Torque, 1, axis=0)
            self.P = np.roll(self.P, 1, axis=0)
            self.cl = np.roll(self.cl, 1, axis=0)
            self.cd = np.roll(self.cd, 1, axis=0)
            self.eta = np.roll(self.eta, 1, axis=0)
            self.alpha_A = np.roll(self.alpha_A, 1, axis=0)
            self.fd = np.roll(self.fd, 1, axis=0)
            self.torque_struts = np.roll(self.torque_struts, 1, axis=0)

            self.acc_x *= 0
            self.acc_y *= 0

            for blade in range(0, self.z_blades):

                # get blade position in radians
                pos_x, pos_y, index_x, index_y = self.get_position(blade, T)

                # Kernel weighted mask W
                self.W = self.get_weights(
                    pos_x, pos_y, index_x, index_y, mesh, self.phi[0, blade], self.chord, self.h)

                # calculate struts correction
                if self.parasitic_torque == 1:
                    self.get_parasitic_torque(
                        pos_x, pos_y, self.phi[0, blade], u, v, blade, self.struts_width)

                # the basic blade element method
                Fx, Fy = self.BEM(blade, u, v, self.W, dt)

                # apply forces as acceleration through the mask on the velocity field
                self.acc_x += self.W * Fx / (self.rho * self.grid_area)
                self.acc_y += self.W * Fy / (self.rho * self.grid_area)

        dudt = dudt + self.acc_x
        dvdt = dvdt + self.acc_y
        
        return dudt, dvdt


