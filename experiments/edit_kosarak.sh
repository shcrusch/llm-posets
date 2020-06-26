basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py kosarak coor > $basedir/kosarak/kosarak_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py kosarak grad_$step > $basedir/kosarak/kosarak_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py kosarak da_$step > $basedir/kosarak/kosara_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py kosarak acc_grad_$mu >$basedir/kosarak/kosarak_acc_grad_$mu.txt &
done

for step in {0.01,0.1,1,10}; do
    python $basedir/edit_l1.py kosarak prox_$step > $basedir/kosarak/regularized/kosarak_prox_$step.txt &
done

python $basedir/edit_l1.py kosarak coor_l1 > $basedir/kosarak/regularized/kosarak_coor_l1.txt &

for step in {1,2,5,10,20}; do
    python $basedir/edit_l1.py kosarak rda_$step > $basedir/kosarak/regularized/kosarak_rda_$step.txt &
done
