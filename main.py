import argparse
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# Modify Visible Devices to match your GPU Count. c.f.) If you have 4 GPUs, the value is '0, 1, 2, 3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# Local Host -- You do not need to modify if you are using one-machine.
os.environ['MASTER_ADDR'] = '127.0.0.1'

# Port -- You do not need to modify if you are using one-machine.
os.environ['MASTER_PORT'] = '10161'

# Define Parser -- If your code already have ArgumentParser(), remove this line.
parser = argparse.ArgumentParser()

# Init_Distribute -- Do not modify.
def init_distribute(rank, opts):
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)

    if opts.rank is not None:
        print(f"Using GPU: {local_gpu_id}")

    torch.distributed.init_process_group(backend='nccl', world_size=opts.ngpus_per_node, rank=opts.rank)
    torch.distributed.barrier()

    print(f'opts: {opts}')

# Train Code -- You should modify here.
def main(rank, opts):
    init_distribute(rank, opts)
    local_gpu_id = opts.gpu
    device = local_gpu_id
    
    # Modify Batch size to fit your GPU
    BATCH_SIZE = 1

    # Modify Epochs to fit your model
    EPOCHS = 100

    # Define your model here.
    model = YOUR_MODEL()
    model.cuda(local_gpu_id)
    model = DistributedDataParallel(model, device_ids=[local_gpu_id], find_unused_parameters=True)

    # Define your Data Loader here.
    dataloader = YOUR_DATA_LOADER()
    sampler = DistributedSampler(dataloader, shuffle=True)
    batch_sampler = torch.utils.data.BatchSampler(sampler, BATCH_SIZE, drop_last=True)
    data_loader = DataLoader(dataset=dataloader, num_workers=opts.threads, batch_sampler=batch_sampler)

    # Insert your left code here. (You do not need to insert codes related to Epochs, Batch Size, Device, Model, Data Loader.)
    for epoch in range(0, EPOCHS):
        sampler.set_epoch(epoch)

        for i, batch in enumerate(data_loader):
            # Insert your Train Code here.
            print(f'Epoch {epoch}\t')

# Main Entry Point
if __name__ == '__main__':
    opts = parser.parse_args()
    
    opts.ngpus_per_node = torch.cuda.device_count()
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.threads = opts.ngpus_per_node * 4

    torch.multiprocessing.spawn(main, args=(opts, ), nprocs=opts.ngpus_per_node, join=True)