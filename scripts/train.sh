CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/gaitstr.yaml --phase train