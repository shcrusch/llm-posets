basedir=/home/hayashi/workspace/llm-posets
srcdir=$basedir/src/python

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py mushroom grad $step 1001 > $basedir/experiments/mushroom/grad_$step.txt 2> $basedir/experiments/mushroom/log/grad_$step.log &
done


python $srcdir/run.py mushroom coor 1001 > $basedir/experiments/mushroom/coor.txt 2> $basedir/experiments/mushroom/log/coor.log &

for mu in {0.97,0.99,0}; do
    python $srcdir/run.py mushroom acc_grad 1 $mu 1001 > $basedir/experiments/mushroom/acc_grad_$mu.txt 2> $basedir/experiments/mushroom/log/acc_grad_$mu.log &
done



