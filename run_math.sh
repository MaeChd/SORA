for adapter_name in sora lora dora; do
    for lr in 2e-4 4e-4 6e-4; do
        for r in 16; do
            bash scripts/train_and_eval_math.sh $adapter_name $lr 5e-3 2 $r 1 32 /home/ft-training_set/math_14k.json "0,1,2,3,4,5,6,7" 
            echo "done with adapter=$adapter_name, lr=$lr, r=$r"
        done
    done
done