
for dataset in "mushroom" "chess" "connect" "retail" "kosarak"
do
    python edit.py $dataset coor > $dataset/${dataset}_coor.txt &
    for step in {0.01,0.1,1,10}
    do
        python edit.py $dataset grad_$step > $dataset/${dataset}_grad_$step.txt &
    done


    for mu in {0,0.95,0.97,0.99};
    do
        python edit.py $dataset acc_grad_$mu >$dataset/${dataset}_acc_grad_$mu.txt &
    done
done