basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python

#for step in {0.01,0.1,1,10}; do
#    python $srcdir/run.py chess grad $step 2001 > $basedir/experiments/chess/grad_$step.txt 2> $basedir/experiments/chess/log/grad_$step.log &
#done

#for step in {0.01,0.1,1,10}; do
#    python $srcdir/run.py chess da $step 2001 > $basedir/experiments/chess/da_$step.txt 2> $basedir/experiments/chess/log/da_$step.log &
#done

#python $srcdir/run.py chess coor 2001 > $basedir/experiments/chess/coor.txt 2> $basedir/experiments/chess/log/coor.log &

#best stepsize is 0.1
for mu in {0.95,0.97,0.99,0}; do
    python $srcdir/run.py chess acc_grad 0.1 2001 $mu > $basedir/experiments/chess/acc_grad_$mu.txt 2> $basedir/experiments/chess/log/acc_grad_$mu.log &
done

#python $srcdir/run.py chess coor_l1 0.001 2001 > $basedir/experiments/chess/regularized/coor_l1.txt 2> $basedir/experiments/chess/regularized/log/coor_l1.log &

#for step in {0.01,0.1,1,10}; do
#    python $srcdir/run.py chess prox $step 0.001 2001 > $basedir/experiments/chess/regularized/prox_$step.txt 2> $basedir/experiments/chess/regularized/log/prox_$step.log &
#done

#for step in {1,2,5,10,20}; do
#    python $srcdir/run.py chess rda $step 0.001 2001 > $basedir/experiments/chess/regularized/rda_$step.txt 2> $basedir/experiments/chess/regularized/log/rda_$step.log &
#done
