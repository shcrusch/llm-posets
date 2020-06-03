basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python

#for step in {0.01,0.1,1,10}; do
#    python $srcdir/run.py mushroom grad $step 1001 > $basedir/experiments/mushroom/grad_$step.txt 2> $basedir/experiments/mushroom/log/grad_$step.log &
#done

#for step in {0.01,0.1,1,10}; do
#    python $srcdir/run.py mushroom da $step 1001 > $basedir/experiments/mushroom/da_$step.txt 2> $basedir/experiments/mushroom/log/da_$step.log &
#done

python $srcdir/run.py mushroom coor 1001 > $basedir/experiments/mushroom/coor.txt 2> $basedir/experiments/mushroom/log/coor.log &

#for mu in {0.97,0.99,0}; do
#    python $srcdir/run.py mushroom acc_grad 1 1001 $mu > $basedir/experiments/mushroom/acc_grad_$mu.txt 2> $basedir/experiments/mushroom/log/acc_grad_$mu.log &
#done


python $srcdir/run.py mushroom coor_l1 0.001 1001 > $basedir/experiments/mushroom/regularized/coor_l1.txt 2> $basedir/experiments/mushroom/regularized/log/coor_l1.log &

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py mushroom prox $step 0.001 1001 > $basedir/experiments/mushroom/regularized/prox_$step.txt 2> $basedir/experiments/mushroom/regularized/log/prox_$step.log &
done

for step in {1,2,5,10,20}; do
    python $srcdir/run.py mushroom rda $step 0.001 1001 > $basedir/experiments/mushroom/regularized/rda_$step.txt 2> $basedir/experiments/mushroom/regularized/log/rda_$step.log &
done

