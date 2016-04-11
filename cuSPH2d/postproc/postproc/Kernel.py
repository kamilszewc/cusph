#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All right reserved by Kamil Szewc, Gdansk 2009

from math import exp, pow
I_PI = 0.31830988618379069

def kernel(q, H, T):
    """kernel(q, H, T) returns value of 2D kernel in point q, where:
        q = |x_1 - x_2|/H
        H - smooth length
        T - type of kernel
    """
    norm = I_PI/(H*H)

    if T == 0:
        if q < 2.0: kern = exp(-q*q)
        else: kern = 0.0
    elif T == 1:
        if q < 2.0: kern = 0.375*q*q - 1.5*q + 1.5
        else: kern = 0.0
    elif T == 2:
        if q <= 1.0: kern = (10.0-15.0*pow(q,2)+7.5*pow(q,3))/7.0
        elif q < 2.0: kern = 2.5*pow(2.0-q, 3)/7.0
        else: kern = 0.0
    elif T == 3:
        if q < 2.0: kern = 1.75*pow(2.0 - q, 4)*(q+0.5)/8.0
        else: kern = 0.0;
    elif T == 4:
        if q < 1.0: kern = 7.0*(pow(3.0-q,5) - 6.0*pow(2.0-q,5) + 15.0*pow(1.0-q,5))/478.0
        elif q < 2.0: kern = 7.0*(pow(3.0-q,5) - 6.0*pow(2.0-q,5))/478.0
        elif q < 3.0: kern = 7.0*pow(3.0-q,5)/478.0
        else: kern = 0.0
    return norm*kern

def grad_of_kernel(x, q, H, T):
    """grad_of_kernel(x, q, H, T) returns value of grad of 2D kernel in point q, where:
        x = x_1 - x_2
        q = |x|/H
        H - smooth length
        T - type of kernel
    """
    norm = I_PI/(H*H*H*H)

    if T == 0:
        if q<2.0: gk = -2.0*x*exp(-q*q)
        else: gk = 0.0
    elif T == 1:
        if q==0.0: gk = 0.0
        elif q<2.0: gk = (0.75*x - 1.5*x/q)
        else: gk = 0.0
    elif T == 2:
        if  q<=1.0: gk = (-30.0+22.5*q)*x/7.0
        elif q<2.0: gk = -7.5*x*pow(2.0-q, 2)/(7.0*q)
        else: gk = 0.0
    elif T == 3:
        if q<2.0: gk = -1.09375*x*pow(2.0-q, 3)
        else: gk = 0.0
    elif T == 4:
        if 0.0 < q < 1.0: gk = -35.0*x*(pow(3.0-q,4) - 6.0*pow(2.0-q,4) + 15.0*pow(1.0-q,4))/(478.0*q)
        elif 1.0 <= q < 2.0: gk = -35.0*x*(pow(3.0-q,4) - 6.0*pow(2.0-q,4))/(478.0*q)
        elif 2.0 <= q < 3.0: gk = -35.0*x*pow(3.0-q,4)/(478.0*q)
        else: gk = 0.0
    return norm*gk
