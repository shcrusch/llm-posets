basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.9,0.7,0.5,0.3,0.1}; do
    python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets grad $step > $basedir/experiments/kosarak/grad_$step.txt 2> $basedir/experiments/kosarak/log/grad_$step.log &
    python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets stoc $step > $basedir/experiments/kosarak/stoc_$step.txt 2> $basedir/experiments/kosarak/log/stoc_$step.log &

done
python $srcdir/run.py $datadir/kosarak.dat $datadir/kosarak.dat_itemsets coor >$basedir/experiments/kosarak/coor.txt 2> $basedir/experiments/kosarak/log/coor.log &

