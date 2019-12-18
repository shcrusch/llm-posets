basedir=/home/hayashi/workspace/tbm-python
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {0.9,0.5,0.1}; do
    python $srcdir/run.py $datadir/mushroom.dat $datadir/mushroom.dat_itemsets grad $step 10001 > $basedir/experiments/mushroom/grad_$step.txt 2> $basedir/experiments/mushroom/log/grad_$step.log &

done

python $srcdir/run.py $datadir/mushroom.dat $datadir/mushroom.dat_itemsets coor 10001 > $basedir/experiments/mushroom/coor.txt 2> $basedir/experiments/mushroom/log/coor.log &



