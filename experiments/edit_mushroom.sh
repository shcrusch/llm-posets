python edit.py mushroom coor > mushroom/mushroom_coor.txt &
for step in {0.01,0.1,1,10}; do
    python edit.py mushroom grad_$step > mushroom/mushroom_grad_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python edit.py mushroom acc_grad_$mu >mushroom/mushroom_acc_grad_$mu.txt &
done

