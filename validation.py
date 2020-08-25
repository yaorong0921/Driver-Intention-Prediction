import torch
from torch.autograd import Variable
import time
import sys
import os

from utils import AverageMeter, calculate_accuracy

best_prec = 0

def val_epoch(epoch, data_loader, model, criterion, optimizer, opt, logger):
    print('Validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    conf_mat = torch.zeros(opt.n_finetune_classes, opt.n_finetune_classes)
    output_file = []

    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(non_blocking=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        ### print out the confusion matrix
        _,pred = torch.max(outputs,1)
        for t,p in zip(targets.view(-1), pred.view(-1)):
            conf_mat[t,p] += 1

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))
    print(conf_mat)
    global best_prec

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    is_best = accuracies.avg > best_prec
    best_prec = max(accuracies.avg, best_prec)
    print('\n The best prec is %.4f' % best_prec)
    if is_best:
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
         }
        save_file_path = os.path.join(opt.result_path,
                            'save_best.pth')
        torch.save(states, save_file_path)

    return losses.avg
