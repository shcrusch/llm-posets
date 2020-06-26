basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py connect grad $step 5001 > $basedir/experiments/connect/grad_$step.txt 2> $basedir/experiments/connect/log/grad_$step.log &
done

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py connect da $step 5001 > $basedir/experiments/connect/da_$step.txt 2> $basedir/experiments/connect/log/da_$step.log &
done

python $srcdir/run.py connect coor 5001 > $basedir/experiments/connect/coor.txt 2> $basedir/experiments/connect/log/coor.log &

#best stepsize is 0.1
for mu in {0.95,0.97,0.99,0}; do
    python $srcdir/run.py connect acc_grad 0.1 5001 $mu > $basedir/experiments/connect/acc_grad_$mu.txt 2> $basedir/experiments/connect/log/acc_grad_$mu.log &
done

#python $srcdir/run.py connect coor_l1 0.001 2001 > $basedir/experiments/connect/regularized/coor_l1.txt 2> $basedir/experiments/connect/regularized/log/coor_l1.log &

#for step in {0.01,0.1,1,10}; do
#    python $srcdir/run.py connect prox $step 0.001 2001 > $basedir/experiments/connect/regularized/prox_$step.txt 2> $basedir/experiments/connect/regularized/log/prox_$step.log &
#done

#for step in {1,2,5,10,20}; do
#    python $srcdir/run.py connect rda $step 0.001 2001 > $basedir/experiments/connect/regularized/rda_$step.txt 2> $basedir/experiments/connect/regularized/log/rda_$step.log &
#done
