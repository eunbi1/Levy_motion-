# Levy_motion-

# Training 
python main.py --alpha 2.0 --beta_min 0.1 --beta_max=20 --n_epochs=1000 --num_workers 0 --datasets "MNIST"

# Sampling 
python main.py --sample --path 'ckpt.pth' --beta_min 0.1 --beta_max 20 --num_steps=1000 --sample_batch_size 64

# Sampling & Training 
python main.py --train_sample 
