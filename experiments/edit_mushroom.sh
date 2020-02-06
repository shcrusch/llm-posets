basedir=/home/hayashi/workspace/tbm-python/experiments2

python $basedir/edit.py mushroom > $basedir/mushroom/mushroom_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/grad_edit.py mushroom grad $step > $basedir/mushroom/mushroom_grad_$step.txt &
done

for mu in {0,0.5,0.7,0.9,0.95,0.97,0.99}; do
    python $basedir/grad_edit.py mushroom acc_grad $mu >$basedir/mushroom/mushroom_acc_grad_$mu.txt &
done

