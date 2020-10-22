basedir=/home/hayashi/workspace/llm-posets
srcdir=$basedir/src/python

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py chess grad $step 1001 > $basedir/experiments/chess/grad_$step.txt 2> $basedir/experiments/chess/log/grad_$step.log &
done



python $srcdir/run.py chess coor 1001 > $basedir/experiments/chess/coor.txt 2> $basedir/experiments/chess/log/coor.log &

#best stepsize is 0.1
for mu in {0.95,0.97,0.99,0}; do
    python $srcdir/run.py chess acc_grad 0.1 $mu 1001 > $basedir/experiments/chess/acc_grad_$mu.txt 2> $basedir/experiments/chess/log/acc_grad_$mu.log &
done
