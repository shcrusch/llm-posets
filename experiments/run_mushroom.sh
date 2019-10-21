basedir=/home/hayashi/workspace/TBM
srcdir=$basedir/src/python
datadir=$basedir/dataset
for step in {1,0.9,0.5,0.1,0.01}; do
    python $srcdir/run.py $datadir/mushroom.dat $datadir/mushroom.dat_itemsets grad $step > $basedir/experiments/mushroom/grad_$step.txt 2> $basedir/experiments/mushroom/log/grad_$step.log &
python $srcdir/run.py $datadir/mushroom.dat $datadir/mushroom.dat_itemsets coor3 > $basedir/experiments/mushroom/coor.txt 2> $basedir/experiments/mushroom/log/coor.log &

done

