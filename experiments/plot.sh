
for dataset in "mushroom" "chess" "connect" "retail" "kosarak"
do
    for type in "epoch" "time"
    do python plot.py $dataset $type &
    done
done