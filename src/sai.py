import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin
import scipy.linalg as splin

def expm_ArnoldiSAI(A, v, t, gamma, tol, max_iter, disp=True, A_lu=None):
    '''
    Shift-and-invert Krylov method to compute exp(-tA)v with tolerance tol, used shift gamma
    and maximum number of Arnoldi iterations max_iter.
    If LU factorization of the matrix I + gamma*A is given in A_lu, then it is used in solving linear systems 
    '''
    n = v.shape[0]
    convergence = np.zeros(max_iter)
    I_gammaA = spsp.eye(n, format="csc") + gamma * A

    if A_lu is None:
        if disp:
            print('Computing sparse LU factorization of the SAI matrix...')
        A_lu = spsplin.splu(I_gammaA)
        if disp:
            print('Done')

    y_all = np.zeros_like(v)
    res_norm_all = np.zeros(v.shape[1])
    for k in range(v.shape[1]):
        V = np.zeros((n, max_iter+1))
        H = np.zeros((max_iter + 1, max_iter))
        beta = np.linalg.norm(v[:, k])
        V[:, 0] = v[:, k] / beta

        for j in range(max_iter):
            w = A_lu.solve(V[:, j]) # Q*( U\( L\(P*V(:,j)) ) );

            for i in range(j+1):
                H[i, j] = w @ V[:, i]
                w = w - H[i, j] * V[:, i]

            H[j+1, j] = np.linalg.norm(w)
            e1 = np.zeros(j+1) 
            e1[0] = 1
            ej = np.zeros(j+1)
            ej[j] = 1
            invH = np.linalg.inv(H[:j+1, :j+1])
            Hjj = (invH - np.eye(j+1)) / gamma
    #         print(Hjj)
            C = np.linalg.norm(I_gammaA.dot(w))
            s = np.array([1/3, 2/3, 1]) * t
            beta_j = np.zeros_like(s)
            for q in range(len(s)):
                u = splin.expm(-s[q] * Hjj) @ e1
                beta_j[q] = C / gamma * (ej @ (invH @ u));

            resnorm = np.linalg.norm(beta_j, np.inf)
            convergence[j] += resnorm
            if disp:
                print('Iteration = {}, resnorm = {}'.format(j+1, resnorm))
            if resnorm <= tol:
                break
            elif j == max_iter:
                print('warning: no convergence within m steps')

            V[:, j+1] = w / H[j+1, j]
#         u = u[:, np.newaxis]
        y = V[:, :j+1] @ (beta * u)
        y_all[:, k] = y
        res_norm_all[k] = resnorm
    return y_all, resnorm, convergence / v.shape[1]

def expm_ArnoldiSAI2_der(A, v, t, gamma, tol, max_iter, disp=True, A_lu=None):
    '''
    Modification of the SAI Krylov method 
    which estimates the dertivative of the residual norm w.r.t. shift value.
    The linear system for gamma' = gamma + 1e-7 is solved with 
    preconditioned Richardson iteration
    '''
    n = v.shape[0]
    V = np.zeros((n, max_iter+1))
    H = np.zeros((max_iter+1, max_iter))
    V2 = np.zeros((n, max_iter+1))
    H2 = np.zeros((max_iter+1, max_iter))

    gamma2 = gamma + 1e-7

    I_gammaA = spsp.eye(n, format="csc") + gamma * A
    if disp:
        print('Computing sparse LU factorization of the SAI matrix...')
    if A_lu is None: 
        A_lu = spsplin.splu(I_gammaA)
    

    beta = np.linalg.norm(v)
    V[:, 0] = v / beta
    V2[:, 0] = v / beta;

    for j in range(max_iter):
        w = A_lu.solve(V[:, j])
        w2 = A_lu.solve(V2[:, j])
        
        res2 = V2[:, j] - (w2 + gamma2 * (A.dot(w2))) # (gamma-gamma2)*A*w2;
        w2 = w2 + A_lu.solve(res2)

        for i in range(j+1):
            H[i, j]  = w @ V[:, i]
            w = w - H[i, j] * V[:, i]
            
            H2[i, j] = w2 @ V2[:, i]
            w2 = w2 - H2[i, j] * V2[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        H2[j + 1, j] = np.linalg.norm(w2)

        e1 = np.zeros((j+1,))
        e1[0] = 1
        ej = np.zeros((j+1,)) 
        ej[j] = 1

        invH = np.linalg.inv(H[:j+1, :j+1])
        Hjj = (invH - np.eye(j+1)) / gamma

        invH2 = np.linalg.inv(H2[:j+1, :j+1])
        H2jj = (invH2 - np.eye(j+1)) / gamma2

        C = np.linalg.norm(I_gammaA.dot(w))
        C2 = np.linalg.norm(w + gamma2 * (A.dot(w)))
        s = np.array([1/3, 2/3, 1]) * t
        beta_j = np.zeros_like(s)
        beta2j = np.zeros_like(s)
        for q in range(s.shape[0]):
            
            u = splin.expm(-s[q] * Hjj) @ e1
            beta_j[q] = C / gamma * (ej @ (invH@ u))

            u2 = splin.expm(-s[q] * H2jj) @ e1
            beta2j[q] = C2 / gamma2 * (ej @ (invH2 @ u2))

        resnorm = np.linalg.norm(beta_j, np.inf)
        resnorm2 = np.linalg.norm(beta2j, np.inf)
        
        deriv = (resnorm2 - resnorm) / (gamma2 - gamma)
        if disp:
            print('j = {}, resnorm = {}'.format(j, resnorm))
        if resnorm <= tol:
            break
        elif j == max_iter:
            print('warning: no convergence within m steps')

        V[:,j + 1] = w / H[j + 1, j]
        V2[:, j + 1] = w2 / H2[j + 1, j]

    y = V[:, :j+1] @ (beta * u)
    return y, deriv