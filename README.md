# oscars-toolbox
A package for helpful general algorithms I've developed. See the PyPI release: https://pypi.org/project/oscars-toolbox/. See also my wesbite, https://oscars47.github.io/.

## Current functions as of latest version:
 ### ```trabbit``` 
 custom gradient descent algorithm to determine optimal params to minimize loss function.
* How to use: specify loss function, function to generate random parameters. Optionally, you can define bounds for the parameters, an initial list of parameters, the total number of iterations you want, the learning rate, the temperature (how often we seek a random solution), the tolerance for convergence (default is 1e-5), the size of the gradient step (default is 1e-5), and a boolean option verbose for whether to print out progress or not (default = True).
 - example pseudocode usage:
   ```
   from oscars_toolbox.trabbit import trabbit
   from functools import partial

   def loss_func(x, args):
    '''your loss code implementation here. optional args argument'''
    return loss

   def rand_gen():
    '''logic to generate random initial params'''
    return rand_ls

   loss_func_p = partial(loss_func, args=args) # use partial to add your arguments to loss function (optional)

   # call trabbit to run SGD
   x_best, loss_best = trabbit(
    loss_func: loss_func_p,
    random_gen: rand_gen,
    bounds: None,
    x0_ls: None,
    num: int = 1000, 
    alpha: float = 0.3,
    temperature: float = 0.1,
    tol: float = 0.00001,
    grad_step: float = 1e-8,
    verbose: bool = True) # alpha is the learning rate, num is total iterations to run, bounds is optional tuple, x0_ls holds list of initial params to try if known, alpha is learning rate, temperature is the fraction of times without updating x_best before trying a new random x0 based on rand_gen(), tol is the tolerance of the loss function before early exit, and grad_step is the size of the epsilon term when approximating the  gradient, verbose is boolean whether to print out progress per iteration or not
   ```

## Updates
### 0.0.4
```trabbit```: 
* renamed the folder 'gradient_descent' -> 'oscars_toolbox' so now you can actually import the package as 'oscars_toolbox'

### 0.0.3
```trabbit```: 
* renamed ```frac``` -> ```temperature```
* added option for ```bounds``` of inputs
* added parameter for ```grad_step```
* set ```verbose = True``` by default.

### 0.0.2
initial release

