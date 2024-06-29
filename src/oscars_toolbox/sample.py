# file showing sample implementation of trabbit algorithm

import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from trabbit import trabbit # if you want to test using the source code, use this line instead
# from oscars_toolbox import trabbit # if you want to test using the package, use this line instead

# ----- generate data ----- #
def gen_data(degree):
    '''Generates data for a polynomial of degree, function of x and y. Returns x, y, z.

    Parameters:
        :degree: degree of polynomial
    Returns:
        :x: x data
        :y: y data
    '''
    # get random polynomial
    coeffs = np.random.randn(degree+1)
    # generate x data
    x = np.linspace(-1, 1, 100)
    # generate y data
    y = np.polyval(coeffs, x)
    return x, y

# ----- loss function ----- #
def loss_func(x):
    '''Loss function for trabbit algorithm. Returns loss.

    Parameters:
        :x: params
    Returns:
        :loss: loss
    '''
    global x_data, y_data
    # get loss
    loss = np.sum((y_data - np.polyval(x, x_data))**2)
    return loss

# ----- minimize loss function ----- #
def random_gen(degree):
    '''Function factor for random gen Generate random params.
    '''
    return np.random.uniform(low=-3, high=3, size=degree+1)

def find_params(x, y, degree):
    '''Uses trabbit algorithm to minimize loss function. Returns params.
    '''
    global x_data, y_data
    # set x and y data
    x_data = x
    y_data = y

    # set bounds
    bounds = [(-3, 3)] * (degree+1)

    random_gen_func = partial(random_gen, degree=degree)
    
    # run trabbit
    x_best, loss_best = trabbit(loss_func, random_gen_func, bounds=bounds, parallel=True)
    return x_best, loss_best


# ----- for visualization ----- #
def plot_data(x,y, x1 = None, y1 = None):
    '''Plots data, with optional second set of data for comparison.

    Parameters:
        :x: x data
        :y: y data
    '''
    plt.figure(figsize=(10,10))
    plt.plot(x,y, label='data')
    if x1 is not None and y1 is not None:
        plt.plot(x1,y1, label='fit')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x,y = gen_data(3)
    x_best, loss_best = find_params(x,y,3)
    print(f'best params: {x_best}')
    print(f'best loss: {loss_best}')
    y_fit = np.polyval(x_best, x)
    plot_data(x,y, x, y_fit)
