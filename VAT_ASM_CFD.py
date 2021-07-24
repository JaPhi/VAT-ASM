import numpy as np
from numba import njit, prange


def meshing(n_elements, radius, grow_rate_x, grow_rate_y, limits_x, limits_y):
    """ returns meshgrid as X and Y. It takes n_elements for the center turbine area and features
    a linear grow rate until it reaches the corresponding min and max domain borders.

    Args:
        n_elements: n elements across turbine area
        radius: turbine radius
        grow_rate_x, grow_rate_y: linear grow rate for mesh
        xmax ,xmin ,ymax, ymin: horizontal and vertical boundary coordinates


    Returns:
        x, y: one-dimesional vector of mesh coordinates
        X, Y: 2D array meshgrid
        nx, ny: number of mesh elements along each axis
        dx, dy: smallest distance between mesh elements in the center area
        grid_area: 2D array of the area around each node

    """

    x = np.linspace(- radius, radius, num=n_elements, endpoint=True)
    y = np.linspace(- radius, radius, num=n_elements, endpoint=True)

    while x[-1] < limits_x[1]:
        element = (x[-1] - x[-2]) * grow_rate_x
        if x[-1] + 2 * element > limits_x[1]:
            x = np.append(x, limits_x[1])
        else:
            x = np.append(x, x[-1] + element)
    x = np.flip(x, axis=0)
    while x[-1] > limits_x[0]:
        element = (x[-1] - x[-2]) * grow_rate_x
        if x[-1] + 2 * element < limits_x[0]:
            x = np.append(x, limits_x[0])
        else:
            x = np.append(x, x[-1] + element)
    x = np.flip(x, axis=0)

    while y[-1] < limits_y[1]:
        element = (y[-1] - y[-2]) * grow_rate_y
        if y[-1] + 2 * element > limits_y[1]:
            y = np.append(y, limits_y[1])
        else:
            y = np.append(y, y[-1] + element)
    y = np.flip(y, axis=0)
    while y[-1] > limits_y[0]:
        element = (y[-1] - y[-2]) * grow_rate_y
        if y[-1] + 2 * element < limits_y[0]:
            y = np.append(y, limits_y[0])
        else:
            y = np.append(y, y[-1] + element)
    y = np.flip(y, axis=0)

    nx = len(x)
    ny = len(y)

    dx = np.min(x[1:-1] - x[0:-2])
    dy = np.min(y[1:-1] - y[0:-2])

    X, Y = np.meshgrid(x, y)

    grid_area = np.ones((ny, nx))
    grid_area[1:-1, 1:-1] = (X[1:-1, 1:-1] - X[1:-1, 0:-2]) * \
        (Y[1:-1, 1:-1] - Y[0:-2, 1:-1])

    return x, y, X, Y, nx, ny, dx, dy, grid_area


@njit(nopython=True, parallel=True)
def solve_pressure_poisson(p, u, v, nx, ny, dt, rho, X, Y):
    """ solves pressure poisson equation on non-uniform grid

    Args:
        p: 2D pressure field from last step
        u, v: velocity components
        nx, ny: mesh element number
        dt: time step
        rho: density
        SC: pre-calculated spatial distances

    Returns:
        p: pressure field
    """
    pn = np.copy(p)
    pn_grad = np.copy(p)
    b = np.copy(p)

    # prepare constant factor
    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):

            h1 = X[i, j] - X[i, j - 1]
            h2 = X[i, j + 1] - X[i, j]
            h3 = Y[i, j] - Y[i - 1, j]
            h4 = Y[i + 1, j] - Y[i, j]

            a = -h2 / (h1 * (h1 + h2))
            y = h1 / (h2 * (h1 + h2))
            c = -h4 / (h3 * (h3 + h4))
            z = h3 / (h4 * (h3 + h4))

            dudx = (a * u[i, j - 1] + (-a - y) * u[i, j] + y * u[i, j + 1])
            dudy = (c * u[i - 1, j] + (-c - z) * u[i, j] + z * u[i + 1, j])
            dvdx = (a * v[i, j - 1] + (-a - y) * v[i, j] + y * v[i, j + 1])
            dvdy = (c * v[i - 1, j] + (-c - z) * v[i, j] + z * v[i + 1, j])

            b[i, j] = rho * (1 / dt * (dudx + dvdy) - dudx **
                             2 - 2 * dudy * dvdx - dvdy**2)

    # iterate for divergence free velocity field (incompressible) @max. 1000 interations.
    # use the old solution of previous steps as a momentum term to accelerate
    # the convergence.
    for q in range(1, 1000):

        for i in prange(0, ny):
            for j in range(0, nx):
                pn_grad[i, j] = p[i, j] - pn[i, j]
                pn[i, j] = p[i, j]

        # gauss-seidel in parallel. check later for proper convergence
        for i in prange(1, ny - 1):
            for j in range(1, nx - 1):

                h1 = X[i, j] - X[i, j - 1]
                h2 = X[i, j + 1] - X[i, j]
                h3 = Y[i, j] - Y[i - 1, j]
                h4 = Y[i + 1, j] - Y[i, j]

                f = 2 / (h1 * (h1 + h2))
                g = 2 / (h2 * (h1 + h2))
                h = 2 / (h3 * (h3 + h4))
                k = 2 / (h4 * (h3 + h4))

                a = pn[i, j - 1]
                c = pn[i, j + 1]
                d = pn[i - 1, j]
                e = pn[i + 1, j]

                p[i, j] = (a * f + c * g + d * h + e *
                           k - b[i, j]) / (f + g + h + k) + 0.6 * pn_grad[i, j]

        # break loop if residual is below threshold. Check every X iterations
        if q % 10 == 0:
            residual = 0
            residual_max = 0
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    residual = abs(pn[i, j] - p[i, j])
                    if residual > residual_max:
                        residual_max = residual
            if residual_max < 1:  # < 1 Pa
                break

    # Pressure boundary condictions
    # left wall
    for i in range(0, ny):
        p[i, 0] = p[i, 1]  # Neumann
    # right wall
    for i in range(0, ny):
        p[i, nx - 1] = p[i, nx - 2]  # Neumann
    # bottom wall
    for j in range(0, nx):
        p[0, j] = 0  # Dirichlet
    # top wall
    for j in range(0, nx):
        p[ny - 1, j] = p[ny - 2, j]  # Neumann

    return p


@njit(nopython=True)
def apply_bc_closed(u, v, v0, nx, ny, use_tower, tower_mask, closed_sides=False):
    """ apply boundary conditions for the closed (walled) flow field (free slip)

    Args:
        p: 2D pressure field from last step
        u, v: velocity components
        nx, ny: mesh element number
        use_tower: user setting if tower is used
        tower_mask: boolean mask to set velocity to zero

    Returns:
        u, v: velocity components
    """

    # left column
    for i in range(0, ny):
        u[i, 0] = 0
        v[i, 0] = v[i, 1]

    # right column
    for i in range(0, ny):
        u[i, -1] = 0
        v[i, -1] = v[i, -2]

    # bottom row
    for j in range(0, nx):
        u[0, j] = u[1, j]
        v[0, j] = v[1, j]  # Neumann

    # top row
    for j in range(0, nx):
        u[-1, j] = u[-2, j]
        v[-1, j] = -v0

    # tower correction
    if use_tower == True:
        u *= tower_mask
        v *= tower_mask

    return u, v


@njit(nopython=True)
def apply_bc(u, v, v0, nx, ny, use_tower, tower_mask, closed_sides=False):
    """ apply boundary conditions for the flow field. The user can decide
    whether the side walls let fluid pass through or behave like a channel.

    Args:
        p: 2D pressure field from last step
        u, v: velocity components
        nx, ny: mesh element number
        use_tower: user setting if tower is used
        tower_mask: boolean mask to set velocity to zero
        closed_sides: sets u velocity component to zero at the walls if true
    Returns:
        u, v: velocity components
    """
    if closed_sides == False:
        # left column
        for i in range(0, ny):
            u[i, 0] = u[i, 1]
            v[i, 0] = v[i, 1]

        # right column
        for i in range(0, ny):
            u[i, -1] = u[i, -2]
            v[i, -1] = v[i, -2]
    else:
        # left column
        for i in range(0, ny):
            u[i, 0] = 0
            v[i, 0] = v[i, 1]

        # right column
        for i in range(0, ny):
            u[i, -1] = 0
            v[i, -1] = v[i, -2]

    # bottom row
    for j in range(0, nx):
        u[0, j] = u[1, j]
        v[0, j] = v[1, j]  # Neumann

    # top row
    for j in range(0, nx):
        u[-1, j] = u[-2, j]
        v[-1, j] = -v0

    # tower correction
    if use_tower == True:
        for i in prange(0, ny):
            for j in range(0, nx):
                u[i, j] *= tower_mask[i, j]
                v[i, j] *= tower_mask[i, j]

    return u, v


@njit(nopython=True, parallel=True)
def solve_momentum(dudt, dvdt, u, v, dx, dy, nx, ny, p, rho, nu, X, Y):
    """ Solves Navier-Stokes momentum equation on non-uniform grid

    Args:
        p: 2D pressure field from last step
        u, v: velocity components
        nx, ny: mesh element number
        use_tower: user setting if tower is used
        tower_mask: boolean mask to set velocity to zero

    Returns:
        u, v: velocity components
    """

    # low order derivatives for the elements next to a boundary
    def first_order(i, j):

        h1 = X[i, j] - X[i, j - 1]
        h2 = X[i, j + 1] - X[i, j]
        h3 = Y[i, j] - Y[i - 1, j]
        h4 = Y[i + 1, j] - Y[i, j]

        a = -h2 / (h1 * (h1 + h2))
        y = h1 / (h2 * (h1 + h2))

        c = -h4 / (h3 * (h3 + h4))
        z = h3 / (h4 * (h3 + h4))

        a2 = 2 / (h1 * (h1 + h2))
        y2 = 2 / (h2 * (h1 + h2))

        b2 = 2 / (h3 * (h3 + h4))
        z2 = 2 / (h4 * (h3 + h4))

        if u[i, j] >= 0:
            dudx = (u[i, j] - u[i, j - 1]) / h1
            dvdx = (v[i, j] - v[i, j - 1]) / h1
        else:
            dudx = (u[i, j + 1] - u[i, j]) / h2
            dvdx = (v[i, j + 1] - v[i, j]) / h2
        if v[i, j] >= 0:
            dudy = (u[i, j] - u[i - 1, j]) / h3
            dvdy = (v[i, j] - v[i - 1, j]) / h3
        else:
            dudy = (u[i + 1, j] - u[i, j]) / h4
            dvdy = (v[i + 1, j] - v[i, j]) / h4

        dpdx = (a * p[i, j - 1] + (-a - y) * p[i, j] + y * p[i, j + 1])
        dpdy = (c * p[i - 1, j] + (-c - z) * p[i, j] + z * p[i + 1, j])

        d2udx = (a2 * u[i, j - 1] + (-a2 - y2) * u[i, j] + y2 * u[i, j + 1])
        d2udy = (b2 * u[i - 1, j] + (-b2 - z2) * u[i, j] + z2 * u[i + 1, j])
        d2vdx = (a2 * v[i, j - 1] + (-a2 - y2) * v[i, j] + y2 * v[i, j + 1])
        d2vdy = (b2 * v[i - 1, j] + (-b2 - z2) * v[i, j] + z2 * v[i + 1, j])

        dudt[i, j] = (-u[i, j] * dudx - v[i, j] * dudy -
                      1 / rho * dpdx + nu * (d2udx + d2udy))
        dvdt[i, j] = (-u[i, j] * dvdx - v[i, j] * dvdy -
                      1 / rho * dpdy + nu * (d2vdx + d2vdy))

    # higher order derivatives for the elements inside the domain
    def second_order(i, j):

        h1 = X[i, j] - X[i, j - 1]
        h11 = X[i, j] - X[i, j - 2]
        h2 = X[i, j + 1] - X[i, j]
        h22 = X[i, j + 2] - X[i, j]

        h3 = Y[i, j] - Y[i - 1, j]
        h33 = Y[i, j] - Y[i - 2, j]
        h4 = Y[i + 1, j] - Y[i, j]
        h44 = Y[i + 2, j] - Y[i, j]

        if u[i, j] >= 0:
            a = -h11 / (h1 * (h11 - h1))
            y = h1 / (h11 * (h11 - h1))
            b = - a - y
            dudx = b * u[i, j] + a * u[i, j - 1] + y * u[i, j - 2]
            dvdx = b * v[i, j] + a * v[i, j - 1] + y * v[i, j - 2]
        else:
            a = -h22 / (h2 * (h2 - h22))
            y = h2 / (h22 * (h2 - h22))
            b = - a - y
            dudx = b * u[i, j] + a * u[i, j + 1] + y * u[i, j + 2]
            dvdx = b * v[i, j] + a * v[i, j + 1] + y * v[i, j + 2]

        if v[i, j] >= 0:
            a = h33 / (-h3 * (-h3 + h33))
            y = h3 / (h33 * (-h3 + h33))
            b = - a - y
            dudy = b * u[i, j] + a * u[i - 1, j] + y * u[i - 2, j]
            dvdy = b * v[i, j] + a * v[i - 1, j] + y * v[i - 2, j]
        else:
            a = h44 / (-h4 * (-h4 + h44))
            y = h4 / (h44 * (-h4 + h44))
            b = - a - y
            dudy = u[i + 2, j] * -y + u[i + 1, j] * -a + u[i, j] * -b
            dvdy = v[i + 2, j] * -y + v[i + 1, j] * -a + v[i, j] * -b

        a = -h2 / (h1 * (h1 + h2))
        y = h1 / (h2 * (h1 + h2))

        c = -h4 / (h3 * (h3 + h4))
        z = h3 / (h4 * (h3 + h4))

        a2 = 2 / (h1 * (h1 + h2))
        y2 = 2 / (h2 * (h1 + h2))

        b2 = 2 / (h3 * (h3 + h4))
        z2 = 2 / (h4 * (h3 + h4))

        dpdx = (a * p[i, j - 1] + (-a - y) * p[i, j] + y * p[i, j + 1])
        dpdy = (c * p[i - 1, j] + (-c - z) * p[i, j] + z * p[i + 1, j])

        d2udx = (a2 * u[i, j - 1] + (-a2 - y2) * u[i, j] + y2 * u[i, j + 1])
        d2udy = (b2 * u[i - 1, j] + (-b2 - z2) * u[i, j] + z2 * u[i + 1, j])
        d2vdx = (a2 * v[i, j - 1] + (-a2 - y2) * v[i, j] + y2 * v[i, j + 1])
        d2vdy = (b2 * v[i - 1, j] + (-b2 - z2) * v[i, j] + z2 * v[i + 1, j])

        dudt[i, j] = (-u[i, j] * dudx - v[i, j] * dudy -
                      1 / rho * dpdx + nu * (d2udx + d2udy))
        dvdt[i, j] = (-u[i, j] * dvdx - v[i, j] * dvdy -
                      1 / rho * dpdy + nu * (d2vdx + d2vdy))

    # left column
    for i in range(1, ny - 1):
        first_order(i, 1)
    # right column
    for i in range(1, ny - 1):
        first_order(i, nx - 2)
    # bottom row
    for j in range(1, nx - 1):
        first_order(1, j)
    # top row
    for j in range(1, nx - 1):
        first_order(ny - 2, j)

    # inner Domain
    for i in prange(2, ny - 2):
        for j in range(2, nx - 2):
            second_order(i, j)

    return dudt, dvdt
