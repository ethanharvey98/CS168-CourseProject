def least_sqaures(X, y, alpha):
    """
    
    """
    n_samples, n_features = X.shape
    P = cvxopt.matrix((X.T@X).astype(np.double))
    q = cvxopt.matrix(-X.T@y)
    A = cvxopt.matrix(np.identity(n_features))
    b = cvxopt.matrix(np.ones(n_features)*alpha)
    cvxopt.solvers.options['show_progress'] = False
    return cvxopt.solvers.qp(P, q, A, b)["x"]

def find_best_alpha(X, y):
    """
    
    """
    alphas = [i**2/100 for i in range(11)]
    kfold = KFold(n_splits=5)
    average_train_scores, average_test_scores = [], []
    for alpha in alphas:
        train_scores, test_scores = [], []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx,:], X[test_idx,:]
            y_train, y_test = y[train_idx], y[test_idx]
            weights = least_sqaures(X_train, y_train, alpha)
            train_scores.append(accuracy_score(y_train, round_predictions(np.rint(X_train@weights),0,3)))
            test_scores.append(accuracy_score(y_test, round_predictions(np.rint(X_test@weights),0,3)))
        average_train_scores.append(np.mean(train_scores))
        average_test_scores.append(np.mean(test_scores))
    return alphas[np.argmax(average_test_scores)]