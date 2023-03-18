import sympy as sym
import numpy as np
import sys
import matplotlib.pyplot as plt


def mesh_uniform(N_e, d, Omega=[0,1], symbolic=False):
    """
    Return a 1D finite element mesh on Omega with N_e elements of
    the polynomial degree d. The elements have uniform length.
    Return vertices (vertices), local vertex to global
    vertex mapping (cells), and local to global degree of freedom
    mapping (dof_map).
    If symbolic is True, the vertices are expressed as rational
    sympy expressions with the symbol h as element length.
    """
    if symbolic:
        h = sym.Symbol('h')  # element length
        dx = h*sym.Rational(1, d)  # node spacing
        vertices = [Omega[0] + i*dx for i in range(N_e + 1)]
    else:
        vertices = np.linspace(Omega[0], Omega[1], N_e + 1).tolist()
    if d == 0:
        dof_map = [[e] for e in range(N_e)]
    else:
        dof_map = [[e*d + i for i in range(d+1)] for e in range(N_e)]
    cells = [[e, e+1] for e in range(N_e)]
    return vertices, cells, dof_map

def element_matrix(phi, Omega_e, symbolic=True, numint=None):
    n = len(phi)
    A_e = sym.zeros(n, n)
    X = sym.Symbol('X')
    if symbolic:
        h = sym.Symbol('h')
    else:
        h = Omega_e[1] - Omega_e[0]
    detJ = h/2  # dx/dX
    if numint is None:
        for r in range(n):
            for s in range(r, n):
                A_e[r,s] = sym.integrate(phi[r]*phi[s]*detJ, (X, -1, 1))
                A_e[s,r] = A_e[r,s]
    else:
        #phi = [sym.lambdify([X], phi[r]) for r in range(n)]
        # Do instead a phi_rj = phi[r].subs(X, Xj) to avoid real numbers
        for r in range(n):
            for s in range(r, n):
                for j in range(len(numint[0])):
                    Xj, wj = numint[0][j], numint[1][j]
                    phi_rj = phi[r].subs(X, Xj)
                    phi_sj = phi[s].subs(X, Xj)
                    A_e[r,s] += phi_rj*phi_sj*detJ*wj
                A_e[s,r] = A_e[r,s]
    return A_e

def element_vector(f, phi, Omega_e, symbolic=True, numint=None):
    n = len(phi)
    b_e = sym.zeros(n, 1)
    # Make f a function of X (via f.subs to avoid real numbers from lambdify)
    X = sym.Symbol('X')
    if symbolic:
        h = sym.Symbol('h')
    else:
        h = Omega_e[1] - Omega_e[0]
    x = (Omega_e[0] + Omega_e[1])/2 + h/2*X  # mapping
    f = f.subs('x', x)
    detJ = h/2
    if numint is None:
        for r in range(n):
            if symbolic:
                I = sym.integrate(f*phi[r]*detJ, (X, -1, 1))
            if not symbolic or isinstance(I, sym.Integral):
                # Ensure h is numerical
                h = Omega_e[1] - Omega_e[0]
                detJ = h/2
                #integrand = sym.lambdify([X], f*phi[r]*detJ, modules='sympy')
                integrand = sym.lambdify([X], f*phi[r]*detJ)
                #integrand = integrand.subs(sym.pi, np.pi)
                # integrand may still contain symbols like sym.pi that
                # prevents numerical evaluation...
                try:
                    I = sym.mpmath.quad(integrand, [-1, 1])
                except Exception as e:
                    print( 'Could not integrate f*phi[r] numerically:')
                    print( e)
                    sys.exit(0)
            b_e[r] = I
    else:
        #phi = [sym.lambdify([X], phi[r]) for r in range(n)]
        # f contains h from the mapping, substitute X with Xj
        # instead of f = sym.lambdify([X], f)
        for r in range(n):
            for j in range(len(numint[0])):
                Xj, wj = numint[0][j], numint[1][j]
                fj = f.subs(X, Xj)
                phi_rj = phi[r].subs(X, Xj)
                b_e[r] += fj*phi_rj*detJ*wj
    return b_e

def exemplify_element_matrix_vector(f, d, symbolic=True, numint=False):
    Omega_e = [0.1, 0.2]
    A_e = element_matrix(phi, Omega_e=Omega_e,
                         symbolic=symbolic, numint=numint)
    integration_msg = """
    Symbolic integration failed, and then numerical integration
    encountered an undefined symbol (because of the symbolic expressions):
    %s"""
    if symbolic:
        h = sym.Symbol('h')
        Omega_e=[1*h, 2*h]
    try:
        b_e = element_vector(f, phi, Omega_e=Omega_e,
                             symbolic=symbolic, numint=numint)
    except NameError as e:
        raise NameError(integration_msg % e)
    print( 'Element matrix:\n', A_e)
    print( 'Element vector:\n', b_e)


def assemble(vertices, cells, dof_map, phi, f,
             symbolic=True, numint=None):
    # import sets
    N_n = len(list(set(np.array(dof_map).ravel())))
    N_e = len(cells)
    if symbolic:
        A = sym.zeros(N_n, N_n)
        b = sym.zeros(N_n, 1)   # note: (N_n, 1) matrix
    else:
        A = np.zeros(N_n, N_n)
        b = np.zeros(N_n)
    for e in range(N_e):
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        A_e = element_matrix(phi[e], Omega_e, symbolic, numint)
        b_e = element_vector(f, phi[e], Omega_e, symbolic, numint)
        #print( 'element', e)
        #print( b_e)
        for r in range(len(dof_map[e])):
            for s in range(len(dof_map[e])):
                A[dof_map[e][r],dof_map[e][s]] += A_e[r,s]
            b[dof_map[e][r]] += b_e[r]
    return A, b

def approximate(f, symbolic=False, d=1, N_e=4, numint=None,
                Omega=[0, 1], filename='tmp'):
    if symbolic:
        if numint == 'Trapezoidal':
            numint = [[sym.S(-1), sym.S(1)], [sym.S(1), sym.S(1)]]  # sympy integers
        elif numint == 'Simpson':
            numint = [[sym.S(-1), sym.S(0), sym.S(1)],
                      [sym.Rational(1,3), sym.Rational(4,3), sym.Rational(1,3)]]
        elif numint == 'Midpoint':
            numint = [[sym.S(0)],  [sym.S(2)]]
        elif numint == 'GaussLegendre2':
            numint = [[-1/sym.sqrt(3), 1/sym.sqrt(3)], [sym.S(1), sym.S(1)]]
        elif numint == 'GaussLegendre3':
            numint = [[-sym.sqrt(sym.Rational(3,5)), 0,
                       sym.sqrt(sym.Rational(3,5))],
                      [sym.Rational(5,9), sym.Rational(8,9),
                       sym.Rational(5,9)]]
        elif numint is not None:
            print( 'Numerical rule %s is not supported' % numint)
            numint = None
    else:
        if numint == 'Trapezoidal':
            numint = [[-1, 1], [1, 1]]
        elif numint == 'Simpson':
            numint = [[-1, 0, 1], [1./3, 4./3, 1./3]]
        elif numint == 'Midpoint':
            numint = [[0], [2]]
        elif numint == 'GaussLegendre2':
            numint = [[-1/sqrt(3), 1/sqrt(3)], [1, 1]]
        elif numint == 'GaussLegendre3':
            numint = [[-sqrt(3./5), 0, sqrt(3./5)],
                      [5./9, 8./9, 5./9]]
        elif numint is not None:
            print( 'Numerical rule %s is not supported' % numint)
            numint = None


    vertices, cells, dof_map = mesh_uniform(N_e, d, Omega, symbolic)

    # phi is a list where phi[e] holds the basis in cell no e
    # (this is required by assemble, which can work with
    # meshes with different types of elements).
    # len(dof_map[e]) is the number of nodes in cell e,
    # and the degree of the polynomial is len(dof_map[e])-1
    phi = [basis(len(dof_map[e])-1) for e in range(N_e)]

    print( 'phi basis (reference element):\n', phi)
    A, b = assemble(vertices, cells, dof_map, phi, f,
                    symbolic=symbolic, numint=numint)

    print( 'cells:', cells)
    print( 'vertices:', vertices)
    print( 'dof_map:', dof_map)
    print( 'A:\n', A)
    print( 'b:\n', b)
    #print( sym.latex(A, mode='plain'))
    #print( sym.latex(b, mode='plain'))

    if symbolic:
        c = A.LUsolve(b)
    else:
        c = np.linalg.solve(A, b)

    print( 'c:\n', c)

    if not symbolic:
        print( 'Plain interpolation/collocation:')
        x = sym.Symbol('x')
        f = sym.lambdify([x], f, modules='numpy')
        try:
            f_at_vertices = [f(xc) for xc in vertices]
            print( f_at_vertices)
        except Exception as e:
            print( 'could not evaluate f numerically:')
            print( e)
    # else: nodes are symbolic so f(nodes[i]) only makes sense
    # in the non-symbolic case

    if not symbolic and filename is not None:
        title = 'P%d, N_e=%d' % (d, N_e)
        if numint is None:
            title += ', exact integration'
        else:
            title += ', integration: %s' % numint
        x_u, u = u_glob(np.asarray(c), vertices, cells, dof_map,
                        resolution_per_element=51)
        x_f = np.linspace(Omega[0], Omega[1], 10001) # mesh for f
        plt.plot(x_u, u, '-',
                 x_f, f(x_f), '--')
        plt.legend(['u', 'f'])
        plt.title(title)
        plt.savefig(filename + '.pdf')
        plt.savefig(filename + '.png')
    return c

def u_glob(U, vertices, cells, dof_map,
           resolution_per_element=51):
    """
    Compute (x, y) coordinates of a curve y = u(x), where u is a
    finite element function: u(x) = sum_i of U_i*phi_i(x).
    Method: Run through each element and compute curve coordinates
    over the element.
    """
    x_patches = []
    u_patches = []
    for e in range(len(cells)):
        Omega_e = [vertices[cells[e][0]], vertices[cells[e][1]]]
        local_nodes = dof_map[e]
        d = len(local_nodes) - 1
        X = np.linspace(-1, 1, resolution_per_element)
        x = affine_mapping(X, Omega_e)
        x_patches.append(x)
        u_element = 0
        for r in range(len(local_nodes)):
            i = local_nodes[r]  # global node number
            u_element += U[i]*phi_r(r, X, d)
        u_patches.append(u_element)
    x = np.concatenate(x_patches)
    u = np.concatenate(u_patches)
    return x, u

# if __name__ == '__main__':
#     import sys
#     from scitools.misc import function_UI
#     cmd = function_UI(
#         [phi_r, u_glob, element_matrix, element_vector,
#          exemplify_element_matrix_vector, assemble, approximate],
#         sys.argv)
#     x = sym.Symbol('x')  # needed in eval when expression f contains x
#     eval(cmd)
