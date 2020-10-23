# llm-posets

# Instruction for running the code

**Environment need to be set up**
1. Python3
2. Install Numpy
3. install matplotlib

**Running the code**

*For run.py*

Dataset files: [mushroom, chess, connect, retail, kosarak] 
Algorithm : [coor(Coordinate Descent), grad(Gradient Descent), acc_grad(Accelerated Gradient Descent)]

Coordinate descent algorithm:
: python run.py {datasetname} coor {number of epoch}

Gradient descent algorithm:
: python run.py {datasetname} grad {stepsize} {number of iterations}

Accelerated gradient descent algorithm:
: python run.py {datasetname} acc_grad {stepsize} {momentum} {number of iterations}

*For edit.py*

Coordinate descent algorithm:
python edit.py {dataset name} coor > {dataset name}/{dataset name}_coor.txt

Gradient descent algorithm:
python edit.py {dataset name} grad_{stepsize} > {dataset name}/{dataset name}_grad_{stepsize}.txt

Accelerated gradient descent algorithm:
python edit.py {dataset name} acc_grad_{} > {dataset name}/{dataset name}_acc_grad.txt

*For plot.py*

"python plot.py args1 args2"

1. args1: dataset name
2. args2: choose to generate time or epoch figures for different types of data.
