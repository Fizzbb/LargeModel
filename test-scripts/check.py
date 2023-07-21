import torch
if torch.cuda.is_available():
    print("cuda device count: {}".format(torch.cuda.device_count()))
