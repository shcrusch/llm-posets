srcdir=/home/hayashi/workspace/tbm-python/src/python
expdir=/home/hayashi/workspace/tbm-python/experiments2
datdir=/home/hayashi/workspace/tbm-python/dataset

for i in {0,1,2,3,4,5,6,7,8,9}; do
    python $srcdir/run.py $datdir/mushroom.dat $datdir/mushroom.dat_itemsets coor 1001 > $expdir/mushroom/coor_rand$i.txt &
done
