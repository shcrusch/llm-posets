basedir=/home/hayashi/workspace/TBM
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.9,0.7,0.5,0.3,0.1,0.01}; do
    python $srcdir/run.py $datadir/retail.dat $datadir/retail.dat_itemsets grad $step > $basedir/experiments/retail/grad_$step.txt 2> $basedir/experiments/retail/log/grad_$step.log &
done
for absd in {4,5}; do
    python $srcdir/run.py $datadir/retail.dat $datadir/retail.dat_itemsets coor $absd >$basedir/experiments/retail/coor_$absd.txt 2> $basedir/experiments/retail/log/coor_$absd.log &
done