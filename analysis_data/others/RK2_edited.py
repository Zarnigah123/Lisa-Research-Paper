"""Created on Sun Mar 19 23:10:25 2023."""

import numpy as np


def dtheta_dt(theta, omega):

    return omega;


def domega_dt(theta, omega):
    g, l = 9.81, 0.1
    return -(g / l) * np.sin(theta)


# Finds value of y for a given x
# using step size h
# and initial value y0 at x0.
def rungeKutta(x0, y0, x, h):

    # Count number of iterations
    # using step size or
    # step height h
    n = round((x - x0) / h);

    # Iterate for number of iterations
    y = y0;

    for i in range(1, n + 1):

        # Apply Runge Kutta Formulas
        # to find next value of y
        k1 = h * dtheta_dt(x0, y[0]);
        k2 = h * dtheta_dt(x0 + 0.5 * h, y[0] + 0.5 * k1)

        l1 = h * domega_dt(x0, y[1])
        l2 = h * domega_dt(x[0] + 0.5 * h, y[1] + 0.5 * l1)

        # Update next value of y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2);

        # Update next value of x
        x0 = x0 + h;

    return y;


# Driver Code
if __name__ == "__main__":

    x0 = 0;
    y = [np.randians(179), 0];
    x = 5;
    h = 1;

    print("y(x) =", rungeKutta(x0, y, x, h));
