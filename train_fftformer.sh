# hidden: 16 → 1024, num_heads: 4 → 64 (모두 순서 유지)

for hidden in 16 32 64 128 256 512 1024; do
  for heads in 4 8 16 32 64; do
    python main.py --epochs 100 --lr 0.0005 --hidden $hidden --batch_size 128 --num_heads $heads --model_name "pretrain_fftformer"
    python main.py --epochs 100 --lr 0.0005 --hidden $hidden --batch_size 128 --num_heads $heads --model_name "fine-tunning_fftformer"
  done
done
