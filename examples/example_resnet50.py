# Usage:
# python3 examples/example_resnet50.py
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import datetime
from IPython import embed

import hotline

# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=0,1,2,3

quick_run = os.environ.get('HOTLINE_QUICK_RUN')
gpu_model = torch.cuda.get_device_name(0)
num_gpus = torch.cuda.device_count()
if quick_run:
    batch_size = 8  # smallest possible quick test
else:
    if 'V100-SXM2-16GB' in gpu_model:
        batch_size = min(128 * num_gpus, 1024)
    elif '3090' in gpu_model:
        batch_size = min(256 * num_gpus, 1024)
    elif '2080' in gpu_model:
        batch_size = min(96 * num_gpus, 1024)
override_batch_size = os.environ.get('HOTLINE_BATCH_SIZE')
if override_batch_size:
    batch_size = int(override_batch_size)

print(f'\n\nbatch_size: {batch_size}\n')

last_time = datetime.datetime.now()
first_time = last_time
print(last_time)

model = models.resnet50(weights=None)  # speedup with no pretrained weights
model = torch.nn.DataParallel(model)
model.cuda()
# cudnn.benchmark = False  # speedup with False

dataset_dir = os.environ.get('HOTLINE_DATASET_DIR')
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])  # this simulates an ImageNet size input
trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                        download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda")
model.train()

quick_run = os.environ.get('HOTLINE_QUICK_RUN')
if quick_run:
    wait = 1
    warmup = 0
    active = 1
else:
    wait = 20
    warmup = 19
    active = 1
max_steps = wait + warmup + active

run_name = 'ResNet50'
metadata = {
    'model': 'ResNet50',
    'dataset': 'ImageNet',
    'batch_size': batch_size,
    'optimizer': 'SGD',
    'runtime': [],
}

if num_gpus > 1:
    run_name = f'{run_name}-{num_gpus}xGPUs'
    metadata['model'] = f"{metadata['model']}-{num_gpus}xGPUs"

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active),
    on_trace_ready=hotline.analyze(
        model,
        dataloader,
        run_name=run_name,
        metadata=metadata,
    ),
    record_shapes=False,
    profile_memory=False,
    with_stack=False
) as p:
    for step, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device=device), data[1].to(device=device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        this_time = datetime.datetime.now()
        tdelta = this_time - last_time
        last_time = this_time
        print(f'step: {step} {tdelta}')
        metadata['runtime'].append(tdelta)

        if step >= max_steps:
            break

        p.step()


this_time = datetime.datetime.now()
tdelta = this_time - first_time
print(f'TOTAL RUNTIME: {tdelta}')
