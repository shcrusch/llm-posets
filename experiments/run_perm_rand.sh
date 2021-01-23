srcdir=../src/python
datdir=../dataset

for i in {0,1,2,3,4,5,6,7,8,9}; do
    python $srcdir/run.py $datdir/mushroom.dat $datdir/mushroom.dat_itemsets coor 1001 > mushroom/coor_rand$i.txt &
done
