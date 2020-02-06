basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py $datadir/connect.dat $datadir/connect.dat_itemsets grad $step 1001 > $basedir/experiments/connect/grad_$step.txt 2> $basedir/experiments/connect/log/grad_$step.log &
done

python $srcdir/run.py $datadir/connect.dat $datadir/connect.dat_itemsets coor 1001 > $basedir/experiments/connect/coor.txt 2> $basedir/experiments/connect/log/coor.log &

#best stepsize is 0.1
for mu in {0.5,0.7,0.9,0.95,0.97,0.99}; do
    python $srcdir/run.py $datadir/connect.dat $datadir/connect.dat_itemsets acc_grad 0.1 1001 $mu > $basedir/experiments/connect/acc_grad_$mu.txt 2> $basedir/experiments/connect/log/acc_grad_$mu.log &
done
