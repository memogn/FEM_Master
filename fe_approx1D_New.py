import sympy as sym
import numpy as np
import sys

def mesh_uniform(N_e, d, Omega=[0,1], symbolic=False):
    """
    Return a 1D finite element mesh on Omega with N_e elements of
    the polynomial degree d. The nodes are uniformly spaced.
    Return nodes (coordinates) and elements (connectivity) lists.
    If symbolic is True, the nodes are expressed as rational
    sympy expressions with the symbol h as element length.
    """
    if symbolic:
        h = sym.Symbol('h')  # element length
        dx = h*sym.Rational(1, d)  # node spacing
        nodes = [Omega[0] + i*dx for i in range(N_e*d + 1)]
    else:
        nodes = np.linspace(Omega[0], Omega[1], N_e*d + 1).tolist()
    elements = [[e*d + i for i in range(d+1)] \
                for e in range(N_e)]
    return nodes, elements


def basis(d, point_distribution='uniform', symbolic=False):
    """
    Return all local basis function phi as functions of the
    local point X in a 1D element with d+1 nodes.
    If symbolic=True, return symbolic expressions, else
    return Python functions of X.
    point_distribution can be 'uniform' or 'Chebyshev'.
    """ 
    X = sym.symbols('X')
    if d == 0:
        phi_sym = [1]
    else:
        if point_distribution == 'uniform':
            if symbolic:
	        # Compute symbolic nodes
                h = sym.Rational(1, d)  # node spacing
                nodes = [2*i*h - 1 for i in range(d+1)]
            else:
                nodes = np.linspace(-1, 1, d+1)
        elif point_distribution == 'Chebyshev':
            # Just numeric nodes
            nodes = Chebyshev_nodes(-1, 1, d)

        phi_sym = [Lagrange_polynomial(X, r, nodes)
                   for r in range(d+1)]
    # Transform to Python functions
    phi_num = [sym.lambdify([X], phi_sym[r], modules='numpy')
               for r in range(d+1)]
    return phi_sym if symbolic else phi_num

def Lagrange_polynomial(x, i, points):
    p = 1
    for k in range(len(points)):
        if k != i:
            p *= (x - points[k])/(points[i] - points[k])
    return p

def element_matrix(phi, Omega_e, symbolic=True):
    n = len(phi)
    A_e = sym.zeros(n, n)
    X = sym.Symbol('X')
    if symbolic:
        h = sym.Symbol('h')
    else:
        h = Omega_e[1] - Omega_e[0]
    detJ = h/2  # dx/dX
    for r in range(n):
        for s in range(r, n):
            A_e[r,s] = sym.integrate(phi[r]*phi[s]*detJ, (X, -1, 1))
            A_e[s,r] = A_e[r,s]
    return A_e

def element_vector(f, phi, Omega_e, symbolic=True):
    n = len(phi)
    b_e = sym.zeros(n, 1)
    # Make f a function of X
    X = sym.Symbol('X')
    if symbolic:
        h = sym.Symbol('h')
    else:
        h = Omega_e[1] - Omega_e[0]
    x = (Omega_e[0] + Omega_e[1])/2 + h/2*X  # mapping
    f = f.subs('x', x)  # substitute mapping formula for x
    detJ = h/2  # dx/dX
    for r in range(n):
        b_e[r] = sym.integrate(f*phi[r]*detJ, (X, -1, 1))
    return b_e

def assemble(nodes, elements, phi, f, symbolic=True):
    N_n, N_e = len(nodes), len(elements)
    if symbolic:
        A = sym.zeros(N_n, N_n)
        b = sym.zeros(N_n, 1)    # note: (N_n, 1) matrix
    else:
        A = np.zeros((N_n, N_n))
        b = np.zeros(N_n)
    for e in range(N_e):
        Omega_e = [nodes[elements[e][0]], nodes[elements[e][-1]]]

        A_e = element_matrix(phi, Omega_e, symbolic)
        b_e = element_vector(f, phi, Omega_e, symbolic)

        for r in range(len(elements[e])):
            for s in range(len(elements[e])):
                A[elements[e][r],elements[e][s]] += A_e[r,s]
            b[elements[e][r]] += b_e[r]
    return A, b

def mesh_uniform(N_e, d, Omega=[0,1], symbolic=False):
    """
    Return a 1D finite element mesh on Omega with N_e elements of
    the polynomial degree d. The nodes are uniformly spaced.
    Return nodes (coordinates) and elements (connectivity) lists.
    If symbolic is True, the nodes are expressed as rational
    sympy expressions with the symbol h as element length.
    """
    if symbolic:
        h = sym.Symbol('h')  # element length
        dx = h*sym.Rational(1, d)  # node spacing
        nodes = [Omega[0] + i*dx for i in range(N_e*d + 1)]
    else:
        nodes = np.linspace(Omega[0], Omega[1], N_e*d + 1).tolist()
    elements = [[e*d + i for i in range(d+1)] \
                for e in range(N_e)]
    return nodes, elements

def mesh_symbolic(N_e, d, Omega=[0,1]):
    """
    Return a 1D finite element mesh on Omega with N_e elements of
    the polynomial degree d. The nodes are uniformly spaced.
    Return nodes (coordinates) and elements (connectivity)
    lists, using symbols for the coordinates (rational expressions
    with h as the uniform element length).
    """
    h = sym.Symbol('h')  # element length
    dx = h*sym.Rational(1, d)  # node spacing
    nodes = [Omega[0] + i*dx for i in range(N_e*d + 1)]
    elements = [[e*d + i for i in range(d+1)] \
                for e in range(N_e)]
    return nodes, elements
