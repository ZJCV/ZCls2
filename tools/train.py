import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from zcls2.config import cfg
from zcls2.data.build import build_data
from zcls2.optim.optimizer.build import build_optimizer
from zcls2.optim.lr_scheduler.build import build_lr_scheduler
from zcls2.model.criterion.build import build_criterion
from zcls2.model.model.build import build_model
from zcls2.engine.trainer import train
from zcls2.engine.infer import validate
from zcls2.util.parser import parse
from zcls2.util.collect_env import collect_env_info
from zcls2.util.checkpoint import save_checkpoint
from zcls2.data.dataset.mp_dataset import MPDataset

from zcls2.util import logging

logger = logging.get_logger(__name__)

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def main():
    global best_prec1, best_prec5, best_epoch, args
    best_prec1 = 0
    best_prec5 = 0
    best_epoch = 0
    args = parse()

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # args.output_dir = 'outputs'
    if args.local_rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.setup_logging(local_rank=args.local_rank, output_dir=args.output_dir)
    logger.info("Environment info:\n" + collect_env_info())
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config))
    if args.config:
        cfg.merge_from_file(args.config)
        with open(args.config, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    logger.info("local_rank: {0}, master_addr: {1}, master_port: {2}".format(
        os.environ['LOCAL_RANK'], os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']))
    logger.info("opt_level = {}".format(args.opt_level))
    logger.info("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32, type(args.keep_batchnorm_fp32)))
    logger.info("loss_scale = {}".format(args.loss_scale, type(args.loss_scale)))

    logger.info("CUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    model = build_model(cfg, memory_format)

    # Scale learning rate based on global batch size
    # args.lr = args.lr * float(args.batch_size * args.world_size) / 256.
    args.lr = args.lr * float(cfg.DATALOADER.TRAIN_BATCH_SIZE * args.world_size) / 256.
    optimizer = build_optimizer(args, model)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = build_criterion(cfg).cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                cfg.TRAIN.START_EPOCH = checkpoint['epoch']
                # args.start_epoch = checkpoint['epoch']
                global best_prec1
                global best_prec5
                global best_epoch
                best_prec1 = checkpoint['best_prec1']
                best_prec5 = checkpoint['best_prec5']
                best_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # # Data loading code
    train_sampler, train_loader, val_loader = build_data(args, cfg, memory_format)

    if args.evaluate:
        validate(args, cfg, val_loader, model, criterion)
        return

    warmup = cfg.LR_SCHEDULER.IS_WARMUP
    warmup_epoch = cfg.LR_SCHEDULER.WARMUP_EPOCH

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCH):
        if isinstance(train_loader.dataset, MPDataset):
            train_loader.dataset.set_epoch(epoch)
        elif args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(args, cfg, train_loader, model, criterion, optimizer, epoch)
        torch.cuda.empty_cache()
        if warmup and epoch < warmup_epoch:
            pass
        else:
            lr_scheduler.step()

        # evaluate on validation set
        prec1, prec5 = validate(args, cfg, val_loader, model, criterion)
        torch.cuda.empty_cache()

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            best_prec5 = prec5
            best_epoch = epoch + 1
        logger.info(' * Best_prec@1 {top1:.3f} Best_prec@5 {top5:.3f} Best_epoch {be}'
                    .format(top1=best_prec1, top5=best_prec5, be=best_epoch))

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'arch': cfg.MODEL.ARCH,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best, output_dir=args.output_dir, filename=f'checkpoint_{epoch}.pth.tar')


if __name__ == '__main__':
    main()
