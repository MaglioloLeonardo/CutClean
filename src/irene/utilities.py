import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        if self.name == "loss":
            self.avg = float('inf')
            self.val = float('inf')
            self.sum = float('inf')

    def update(self, val, n=1):
        self.val = float(val)
        if self.count == 0:
            # First update, just set the values directly
            self.sum = float(val) * float(n)
        else:
            # Subsequent updates, accumulate
            self.sum += float(val) * float(n)
        self.count += float(n)
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module[1].register_forward_hook(self.hook_fn)
			self.name = module[0]
		else:
			self.hook = module[1].register_backward_hook(self.hook_fn)
			self.name = module[0]
	def hook_fn(self, module, input, output):
		self.input = input
		self.output = output
	def close(self):
		self.hook.remove()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        num_classes = output.size(1)
        if target.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError(
                f"Expected target to contain integer class indices but got dtype {target.dtype}."
            )

        target_min = int(target.min().item())
        target_max = int(target.max().item())
        if target_min < 0 or target_max >= num_classes:
            raise ValueError(
                "Target contains out-of-range class indices for the model output "
                f"(found min={target_min}, max={target_max}, num_classes={num_classes})."
            )

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
