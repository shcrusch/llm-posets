srcdir=../src/python

for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py retail grad $step 1001 > retail/grad_$step.txt 2> retail/log/grad_$step.log &
done


python $srcdir/run.py retail coor 1001 >retail/coor.txt 2> retail/log/coor.log &

#best stepsize is 1
for mu in {0.95,0.97,0.99,0}; do
    python $srcdir/run.py retail acc_grad 1 $mu 1001 > retail/acc_grad_$mu.txt 2> retail/log/acc_grad_$mu.log &
done

