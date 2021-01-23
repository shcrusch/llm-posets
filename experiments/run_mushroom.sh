
srcdir=$basedir/src/python

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py mushroom grad $step 101 > mushroom/grad_$step.txt 2> mushroom/log/grad_$step.log &
done


python $srcdir/run.py mushroom coor 101 > mushroom/coor.txt 2> mushroom/log/coor.log &

for mu in {0.97,0.99,0}; do
    python $srcdir/run.py mushroom acc_grad 1 $mu 101 > mushroom/acc_grad_$mu.txt 2> mushroom/log/acc_grad_$mu.log &
done



