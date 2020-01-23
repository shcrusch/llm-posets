basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.01,0.1,1,10}; do
    python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets grad $step 1001 > $basedir/experiments/kosarak/grad_$step.txt 2> $basedir/experiments/kosarak/log/grad_$step.log &
done

python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets coor 1001 >$basedir/experiments/kosarak/coor.txt 2> $basedir/experiments/kosarak/log/coor.log &

