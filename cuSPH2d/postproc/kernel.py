#!/usr/bin/env python
from numpy import *
from matplotlib.pyplot import *

def kernel_0(q, h):
    if q < 2.0:
        return 0.21875 * pow(2.0 - q, 4) * (q + 0.5) / (h * h * 3.1415)
    else:
        return 0.0

def grad_of_kernel_0(x, q, h):
    if q < 2.0:
        return -1.09375 * x * pow(2.0 - q, 3) / (3.1415 * h * h * h * h)
    else:
        return 0.0

def kernel_1(q, h):
    value = 0.0

    if q <= 2.0:
        value += pow(3.0 - 1.5 * q, 5)
    if q <= 4.0/3.0:
        value -= 6.0 * pow(2.0 - 1.5 * q, 5)
    if q <= 2.0/3.0:
        value += 15.0 * pow(1.0 - 1.5 * q, 5)

    return 63.0 * value / (478.0 * 3.1415 * h * h * 4.0)

def grad_of_kernel_1(x, q, h):
    value = 0.0

    if q <= 2.0:
        value += 7.5 * pow(3.0 - 1.5 * q, 4) * x / q
    if q <= 4.0/3.0:
        value -= 6.0 * 7.5 * pow(2.0 - 1.5 * q, 4) * x / q
    if q <= 2.0/3.0:
        value += 15.0 * 7.5 * pow(1.0 - 1.5 * q, 4) * x / q

    return 63.0 * value / (478.0 * 3.1415 * h * h * h * h * 4.0)

figure()
x = linspace(-3.0, 3.0, 500)
y1 = [kernel_1(abs(r), 1.0) for r in x]
z1 = [grad_of_kernel_1(r, abs(r), 1.0) for r in x]
plot(x, y1)
plot(x, z1)

figure()
x = linspace(-3.0, 3.0, 500)
y0 = [kernel_0(abs(r), 1.0) for r in x]
z0 = [grad_of_kernel_0(r, abs(r), 1.0) for r in x]
plot(x, y0)
plot(x, z0)

show()
