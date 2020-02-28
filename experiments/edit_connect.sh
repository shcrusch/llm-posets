basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py connect > $basedir/connect/connect_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/grad_edit.py connect grad $step > $basedir/connect/connect_grad_$step.txt &
done

for mu in {0,0.5,0.7,0.9,0.95,0.97,0.99}; do
    python $basedir/grad_edit.py connect acc_grad $mu >$basedir/connect/connect_acc_grad_$mu.txt &
done
