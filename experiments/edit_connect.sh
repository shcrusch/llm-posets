python edit.py connect coor > connect/connect_coor.txt &
for step in {0.01,0.1,1,10}; do
    python edit.py connect grad_$step > connect/connect_grad_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python edit.py connect acc_grad_$mu >connect/connect_acc_grad_$mu.txt &
done


