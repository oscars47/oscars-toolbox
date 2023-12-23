# oscars-toolbox
A package for helpful general algorithms I've developed. See also my wesbite, https://oscars47.github.io/.

## Current functions as of version 0.0.3:
 # ```trabbit``` 
 custom gradient descent algorithm to determine optimal params to minimize loss function.
* How to use: specify loss function, function to generate random parameters. Optionally, you can define bounds for the parameters, an initial list of parameters, the total number of iterations you want, the learning rate, the temperature (how often we seek a random solution), the tolerance for convergence (default is 1e-5), the size of the gradient step (default is 1e-5), and a boolean option verbose for whether to print out progress or not (default = True).

## Updates
# 0.0.4
```trabbit```: 
* renamed the folder 'gradient_descent' -> 'oscars_toolbox' so now you can actually import the package as 'oscars_toolbox'

# 0.0.3
```trabbit```: 
* renamed ```frac``` -> ```temperature```
* added option for ```bounds``` of inputs
* added parameter for ```grad_step```
* set ```verbose = True``` by default.

# 0.0.2
initial release

