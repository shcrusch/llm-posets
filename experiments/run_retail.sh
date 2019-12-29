basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.01,0.1,0.3,0.5,0.7,0.9}; do
    python $srcdir/run.py $datadir/retail.dat $datadir/retail.dat_itemsets grad $step > $basedir/experiments/retail/grad_$step.txt 2> $basedir/experiments/retail/log/grad_$step.log &

done

python $srcdir/run.py $datadir/retail.dat $datadir/retail.dat_itemsets coor >$basedir/experiments/retail/coor.txt 2> $basedir/experiments/retail/log/coor.log &

