basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py chess coor > $basedir/chess/chess_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py chess grad_$step > $basedir/chess/chess_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py chess grad_$step > $basedir/chess/chess_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py chess acc_grad_$mu >$basedir/chess/chess_acc_grad_$mu.txt &
done
