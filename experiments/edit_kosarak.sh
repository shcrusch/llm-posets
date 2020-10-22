basedir=/home/hayashi/workspace/llm-posets/experiments

python $basedir/edit.py kosarak coor > $basedir/kosarak/kosarak_coor.txt &
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py kosarak grad_$step > $basedir/kosarak/kosarak_grad_$step.txt &
done
for step in {0.01,0.1,1,10}; do
    python $basedir/edit.py kosarak da_$step > $basedir/kosarak/kosara_da_$step.txt &
done

for mu in {0,0.95,0.97,0.99}; do
    python $basedir/edit.py kosarak acc_grad_$mu >$basedir/kosarak/kosarak_acc_grad_$mu.txt &
done

