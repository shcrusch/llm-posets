# Log-linear Model on Posets

# Instruction for running the code

## Environment need to be set up

1. Python3
2. Install Numpy
3. Install matplotlib

## Running the code
### For run.py
Dataset files : mushroom, chess, connect, retail, kosarak

Algorithms : Coordinate Descent, Gradient Descent, Accelerated Gradient Descent

* Coordinate Descent : 
```sh
python run.py {datasetname} coor {number of epoch}
```

* Gradient Descent : 
```sh
python run.py {datasetname} grad {stepsize} {number of iterations}
```

* Accelerated Gradient Descent : 

```sh
python run.py {datasetname} acc_grad {stepsize} {momentum} {number of iterations}
```

Example:
```sh
python run.py mushroom coor 5
```
The output is as follows
```sh
0 : KL divergence:  0.0030725766189352  time : 0.02
1 : KL divergence:  0.0025266243106774  time : 0.10
2 : KL divergence:  0.0022919382698259  time : 0.18
3 : KL divergence:  0.0021709679956974  time : 0.26
4 : KL divergence:  0.0020972542120266  time : 0.34
```
.
### For edit.py

* Coordinate Descent : 
```sh
python edit.py {dataset name} coor > {dataset name}/{dataset name}_coor.txt
```
* Gradient Descent : 
```sh
python edit.py {dataset name} grad_{stepsize} > {dataset name}/{dataset name}_grad_{stepsize}.txt
```
* Accelerated Gradient Descent : 
```sh
python edit.py {dataset name} acc_grad_{momentum} > {dataset name}/{dataset name}_acc_grad.txt
```
### For plot.py

```sh
python plot.py args1 args2
```
1. args1: dataset name
2. args2: choose to generate time or epoch figures for different types of data.
