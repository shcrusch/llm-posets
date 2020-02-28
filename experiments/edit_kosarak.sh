basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py kosarak > $basedir/kosarak/kosarak_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/grad_edit.py kosarak grad $step > $basedir/kosarak/kosarak_grad_$step.txt &
done

for mu in {0,0.5,0.7,0.9,0.95,0.97,0.99}; do
    python $basedir/grad_edit.py kosarak acc_grad $mu >$basedir/kosarak/kosarak_acc_grad_$mu.txt &
done
