basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.9,0.7,0.5,0.3,0.1}; do
    python $srcdir/run.py $datadir/retail.dat $datadir/retail.dat_itemsets grad $step > $basedir/experiments/retail/grad_$step.txt 2> $basedir/experiments/retail/log/grad_$step.log &
    python $srcdir/run.py $datadir/retail.dat $datadir/retail.dat_itemsets stoc $step > $basedir/experiments/retail/stoc_$step.txt 2> $basedir/experiments/retail/log/stoc_$step.log &

done

python $srcdir/run.py $datadir/retail.dat $datadir/retail.dat_itemsets coor >$basedir/experiments/retail/coor.txt 2> $basedir/experiments/retail/log/coor.log &

