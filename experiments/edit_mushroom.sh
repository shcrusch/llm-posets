basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py mushroom coor > $basedir/mushroom/mushroom_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py mushroom grad_$step > $basedir/mushroom/mushroom_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py mushroom grad_$step > $basedir/mushroom/mushroom_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py mushroom acc_grad_$mu >$basedir/mushroom/mushroom_acc_grad_$mu.txt &
done

for step in {0.01,0.1,1,10}; do
    python $basedir/edit_l1.py mushroom prox_$step > $basedir/mushroom/regularized/mushroom_prox_$step.txt &
done

python $basedir/edit_l1.py mushroom coor_l1 > $basedir/mushroom/regularized/mushroom_coor_l1.txt &

for step in {1,2,5,10,20}; do
   python $basedir/edit_l1.py mushroom rda_$step > $basedir/mushroom/regularized/mushroom_rda_$step.txt &
done
