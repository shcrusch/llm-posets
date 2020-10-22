basedir=/home/hayashi/workspace/llm-posets/experiments

python $basedir/edit.py connect coor > $basedir/connect/connect_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py connect grad_$step > $basedir/connect/connect_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py connect da_$step > $basedir/connect/connect_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py connect acc_grad_$mu >$basedir/connect/connect_acc_grad_$mu.txt &
done


