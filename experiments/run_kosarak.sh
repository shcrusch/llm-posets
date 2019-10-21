basedir=/home/hayashi/workspace/TBM
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.9,0.7,0.5,0.3,0.1,0.01}; do
    python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets grad $step > $basedir/experiments/kosarak/grad_$step.txt 2> $basedir/experiments/kosarak/log/grad_$step.log &
python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets coor3 >$basedir/experiments/kosarak/coor.txt 2> $basedir/experiments/kosarak/log/coor.log &
done
