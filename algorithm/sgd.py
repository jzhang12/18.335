def sgd(A, x, b, lr = 1e-4, tol = 1e-5, nmax = 1e6):
    r = b-A.dot(x)
    iter_num = 0
    res = [linalg.norm(r)/linalg.norm(b)]
    while iter_num < nmax:
        if res[iter_num] < tol:
            break
        iter_num += 1
        x = x + 2*lr*np.dot(np.transpose(A),r)
        r = b-A.dot(x)
        res.append(linalg.norm(r)/linalg.norm(b))
    return res, n, x