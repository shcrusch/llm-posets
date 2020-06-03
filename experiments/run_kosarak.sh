basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py kosarak grad $step 1001 > $basedir/experiments/kosarak/grad_$step.txt 2> $basedir/experiments/kosarak/log/grad_$step.log &
done
for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py kosarak da $step 1001 > $basedir/experiments/kosarak/da_$step.txt 2> $basedir/experiments/kosarak/log/da_$step.log &
done
python $srcdir/run.py kosarak coor 1001 >$basedir/experiments/kosarak/coor.txt 2> $basedir/experiments/kosarak/log/coor.log &

#best stepsize is 1
for mu in {0.95,0.97,0.99,0}; do
    python $srcdir/run.py kosarak acc_grad 1 1001 $mu > $basedir/experiments/kosarak/acc_grad_$mu.txt 2> $basedir/experiments/kosarak/log/acc_grad_$mu.log &
done
