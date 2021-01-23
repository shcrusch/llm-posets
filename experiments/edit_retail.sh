
python edit.py retail coor > retail/retail_coor.txt &
for step in {0.01,0.1,1,10}; do
    python edit.py retail grad_$step > retail/retail_grad_$step.txt &
done


for mu in {0,0.95,0.97,0.99}; do
    python edit.py retail acc_grad_$mu >retail/retail_acc_grad_$mu.txt &
done

