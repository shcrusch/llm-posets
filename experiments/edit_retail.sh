basedir=/home/hayashi/workspace/tbm-python/experiments

python $basedir/edit.py retail coor > $basedir/retail/retail_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py retail grad_$step > $basedir/retail/retail_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py retail da_$step > $basedir/retail/retail_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py retail acc_grad_$mu >$basedir/retail/retail_acc_grad_$mu.txt &
done

#for step in {0.01,0.1,1,10}; do
#    python $basedir/edit_l1.py retail prox_$step > $basedir/retail/regularized/retail_prox_$step.txt &
#done

#python $basedir/edit_l1.py retail coor_l1 > $basedir/retail/regularized/retail_coor_l1.txt &

#for step in {1,2,5,10,20}; do
#    python $basedir/edit_l1.py retail rda_$step > $basedir/retail/regularized/retail_rda_$step.txt &
#done
