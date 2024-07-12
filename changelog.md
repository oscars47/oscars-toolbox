## Updates

### 0.1.7
* updated plot for implement_xgb

### 0.1.6
* fixed various issues, notably the ROC-AUC calc for torch models and using `time.perf_counter()` instead of `time.time()` for increased time resolution

### 0.1.5
* added probability vectors for correct roc-auc usage
* changed default extra param argument to true

### 0.1.4
* added `implement_xgb.evaluate_xgb```

### 0.1.3
* added ```implement_xgb.py``` and added the function ```evaluate_xgb```

### 0.1.2
* added timing data for prediction within `implement_torch.evaluate`.

### 0.1.1
* within `implement_torch` added ROC AUC score

### 0.1.0
* added ```implement_torch```, ```torch_models```

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