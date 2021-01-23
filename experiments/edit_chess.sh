python edit.py chess coor > chess/chess_coor.txt &
for step in {0.01,0.1,1,10}; do
    python edit.py chess grad_$step > chess/chess_grad_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python edit.py chess acc_grad_$mu >chess/chess_acc_grad_$mu.txt &
done

