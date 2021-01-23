
python edit.py kosarak coor > kosarak/kosarak_coor.txt &
for step in {0.01,0.1,1,10}; do
    python edit.py kosarak grad_$step > kosarak/kosarak_grad_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python edit.py kosarak acc_grad_$mu >kosarak/kosarak_acc_grad_$mu.txt &
done

