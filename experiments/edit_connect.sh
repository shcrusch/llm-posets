basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py connect coor > $basedir/connect/connect_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py connect grad_$step > $basedir/connect/connect_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py connect da_$step > $basedir/connect/connect_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py connect acc_grad_$mu >$basedir/connect/connect_acc_grad_$mu.txt &
done

#for step in {0.01,0.1,1,10}; do
#    python $basedir/edit_l1.py connect prox_$step > $basedir/connect/regularized/connect_prox_$step.txt &
#done

#python $basedir/edit_l1.py connect coor_l1 > $basedir/connect/regularized/connect_coor_l1.txt &

#for step in {1,2,5,10,20}; do
#    python $basedir/edit_l1.py connect rda_$step > $basedir/connect/regularized/connect_rda_$step.txt &
#done
