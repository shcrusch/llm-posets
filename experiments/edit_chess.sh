basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py chess coor > $basedir/chess/chess_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py chess grad_$step > $basedir/chess/chess_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py chess da_$step > $basedir/chess/chess_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py chess acc_grad_$mu >$basedir/chess/chess_acc_grad_$mu.txt &
done

for step in {0.01,0.1,1,10}; do
    python $basedir/edit_l1.py chess prox_$step > $basedir/chess/regularized/chess_prox_$step.txt &
done

python $basedir/edit_l1.py chess coor_l1 > $basedir/chess/regularized/chess_coor_l1.txt &

for step in {1,2,5,10,20}; do
    python $basedir/edit_l1.py chess rda_$step > $basedir/chess/regularized/chess_rda_$step.txt &
done
