basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.01,,0.1,1,10}; do
    python $srcdir/run.py $datadir/chess.dat $datadir/chess.dat_itemsets grad $step 1001 > $basedir/experiments/chess/Grad_$step.txt 2> $basedir/experiments/chess/log/Grad_$step.log &

done
python $srcdir/run.py $datadir/chess.dat $datadir/chess.dat_itemsets coor 1001 > $basedir/experiments/chess/Coor.txt 2> $basedir/experiments/chess/log/Coor.log &



