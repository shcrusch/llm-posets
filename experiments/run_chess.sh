basedir=/home/hayashi/workspace/TBM
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.9,0.7,0.5,0.3,0.1,0.01}; do
    python $srcdir/run.py $datadir/chess.dat $datadir/chess.dat_itemsets grad $step > $basedir/experiments/chess/grad_$step.txt 2> $basedir/experiments/chess/log/grad_$step.log &
done

