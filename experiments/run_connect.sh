basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.5,0.3,0.1}; do
    python $srcdir/run.py $datadir/connect.dat $datadir/connect.dat_itemsets grad $step 2001 > $basedir/experiments/connect/grad_$step.txt 2> $basedir/experiments/connect/log/grad_$step.log &

done
python $srcdir/run.py $datadir/connect.dat $datadir/connect.dat_itemsets coor 2001 > $basedir/experiments/connect/coor.txt 2> $basedir/experiments/connect/log/coor.log &



