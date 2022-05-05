def relu(predictions):
    return np.maximum(0, predictions)

def construct_D(A, theta):
    # D is a diagonal matrix and its ith diagonal entry is computed as 1 if A[i].T@theta >= 0, 0 otherwise
    n_samples, n_features = A.shape
    d = np.array([1 if A[i].T@theta >= 0 else 0 for i in range(n_samples)])
    D = np.diag(d)
    return D

def construct_Dc(A, thetas):
    n_features, n_neurons = thetas.shape
    n_samples, n_features = A.shape
    Dcs = np.ones((n_samples, n_samples))
    for neuron_index in range(n_neurons):
        D = construct_D(A, thetas[:,neuron_index])
        Dc = np.identity(len(D)) - D
        Dcs *= Dc
    return Dcs

def residual(A, y, thetas):
    return np.sum(relu(A@thetas), axis=1)-y

def single_neuron_cost(A, y, theta):
    n_samples, n_features = A.shape
    predictions = relu(A@theta)
    cost = (np.linalg.norm(predictions, ord=2)**2)-(2*y.T@A@theta)+(np.linalg.norm(y, ord=2)**2)
    return cost

def single_neuron_training(A, y, theta, max_it=10_000):
    it = 0
    alpha = 0.01
    beta = 0.5
    n_samples, n_features = A.shape
    predictions = relu(A@theta)
    D = construct_D(A, theta)
    gradient = -A.T@D@(predictions-y)
    while np.linalg.norm(gradient)>1e-8 and it<max_it:
        predictions = relu(A@theta)
        D = construct_D(A, theta)
        gradient = -A.T@D@(predictions-y)
        # bracktracking line search
        learning_rate = 1
        while single_neuron_cost(A, y, theta + (learning_rate*gradient)) > single_neuron_cost(A, y, theta) + alpha*learning_rate*(-gradient.T@gradient):
            learning_rate = beta * learning_rate
        theta = theta + (learning_rate*gradient)
        # iterations
        it = it + 1
        if it==max_it: print("Max iterations reached, function did not converge")
    return theta

def multineuron_cost(A, y, thetas):
    n_samples, n_features = A.shape
    predictions = relu(A@thetas)
    Dc = construct_Dc(A, thetas)
    # transpose predictions for element wise subtraction
    cost = np.sum(np.linalg.norm(predictions.T-y, ord=2)**2)-2*y.T@Dc@A@np.sum(thetas, axis=1)
    return cost

def multineuron_training(A, y, thetas, max_it=1_000):
    it = 0
    n_features, n_neurons = thetas.shape
    n_samples, n_features = A.shape
    gradients = np.ones(n_neurons)
    cost_indicies = np.zeros(n_neurons)
    while np.linalg.norm(gradients)>1e-8 and it<max_it:
        for neuron_index in range(n_neurons):
            D = construct_D(A, thetas[:,neuron_index])
            r = residual(A, y, thetas)
            Dc = construct_Dc(A, thetas)
            gradient = A.T@D@r+A.T@Dc@r
            # line search
            learning_rates = np.logspace(-99,0,num=100)
            new_thetas = np.array([thetas for i in range(len(learning_rates))])
            for index, learning_rate in enumerate(learning_rates): new_thetas[index][:,neuron_index] = new_thetas[index][:,neuron_index] - learning_rate*gradient
            costs = np.array([multineuron_cost(A, y, new_theta) for new_theta in new_thetas])
            learning_rate = learning_rates[np.argmin(costs)]
            thetas = new_thetas[np.argmin(costs)]
            gradients[neuron_index] = np.linalg.norm(gradient)
            cost_indicies[neuron_index] = np.argmin(costs)
        # iterations
        it = it + 1
        if it==max_it: print("Max iterations reached, function did not converge")
        print(cost_indicies)
        if (cost_indicies == 0).all():
            print("Reached global minimum")
            return thetas
    return thetas
