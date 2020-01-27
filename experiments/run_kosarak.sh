basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets grad $step 1001 > $basedir/experiments/kosarak/grad_$step.txt 2> $basedir/experiments/kosarak/log/grad_$step.log &
done

#python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets coor 1001 >$basedir/experiments/kosarak/coor.txt 2> $basedir/experiments/kosarak/log/coor.log &

#for mu in {0.5,0.7,0.9,0.95,0.97,0.99}; do
#    python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets acc_grad 1 1001 $mu > $basedir/experiments/kosarak/acc_grad_$mu.txt 2> $basedir/experiments/kosarak/log/acc_grad_$mu.log &
#done
