basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.01,0.1,0.3,0.5,0.7,0.9}; do
    python $srcdir/run.py $datadir/chess.dat $datadir/chess.dat_itemsets grad $step 2001 > $basedir/experiments/chess/grad_$step.txt 2> $basedir/experiments/chess/log/grad_$step.log &
done
python $srcdir/run.py $datadir/chess.dat $datadir/chess.dat_itemsets coor 2001 > $basedir/experiments/chess/coor.txt 2> $basedir/experiments/chess/log/coor.log &
