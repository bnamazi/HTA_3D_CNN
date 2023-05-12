import time
import os
import copy
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, utils, models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

# from skimage import io, transform
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import wandb

from dataloaders import ImageDataset, VideoDataset
from data_augmentation import transform_train, transform_val
from models import CNN_timm, CNN_3D
from losses import LabelSmoothingCrossEntropy, EMD


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir",
    type=str,
    default="D:/data/Lap Chole/HTA_images",
    help="path to training directory ",
)
parser.add_argument(
    "--test_dir",
    type=str,
    default="D:/data/Lap Chole/HTA_images",
    help="path to test directory",
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="batch size (default: 32)"
)
parser.add_argument(
    "--num_epochs", type=int, default=100, help="number of epochs (default: 50)"
)
parser.add_argument("--device", type=str, default="cuda:1")
parser.add_argument(
    "--positional_encoding", type=bool, default=False, help="encoding the frame number"
)
parser.add_argument(
    "--emd_loss",
    type=bool,
    default=False,
    help="earth movers distance as the main loss function",
)
parser.add_argument(
    "--emd_reg",
    type=bool,
    default=False,
    help="earth movers distance as a second loss function",
)
parser.add_argument("--LR", type=float, default=0.001, help="initial learning rate")
parser.add_argument(
    "--decay_rate", type=float, default=0.7, help="rate of learning rate decay"
)
parser.add_argument(
    "--decay_step",
    type=int,
    default=5,
    help="number of steps before learning rate decay",
)
parser.add_argument(
    "--saved_path",
    type=str,
    default="./runs/x3d_phase_p.pth",
    help="path to the saved model",
)
parser.add_argument(
    "--summary_path",
    type=str,
    default="./runs/x3d_phase_p",
    help="path to save tensorboard summaries",
)
parser.add_argument(
    "--loss_ratio",
    type=float,
    default=0.1,
    help="the weight of EMD loss when added to CE",
)


args = parser.parse_args()
# with torch.profiler.profile() as profiler:
#     pass

torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True


device = torch.device(args.device)

num_classes = 5
num_workers = 32


model_saved_path = args.saved_path
model_summary_path = args.summary_path


def run():
    wandb.init(project="HTA", entity="bnamazi", settings=wandb.Settings(code_dir="."))

    wandb.run.log_code(
        ".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb")
    )

    config = wandb.config
    config.learning_rate = args.LR
    config.model = "x3d_phase_p"
    config.batch_size = args.batch_size
    config.positional_encoding = args.positional_encoding
    # torch.multiprocessing.freeze_support()
    # writer = SummaryWriter(log_dir=model_summary_path)

    num_epochs_stop = 15
    epochs_no_improve = 0
    early_stop = False

    train_dataset = VideoDataset(
        data_dir=args.train_dir,
        transform=transform_train,
        positional_encoding=args.positional_encoding,
        train=True,
        split="train",
        clip_length=16,
    )
    test_dataset = VideoDataset(
        data_dir=args.test_dir,
        transform=transform_val,
        positional_encoding=args.positional_encoding,
        train=False,
        split="val",
    )
    validation_dataset = VideoDataset(
        data_dir=args.test_dir,
        transform=transform_val,
        positional_encoding=args.positional_encoding,
        train=False,
        split="val",
        clip_length=16,
    )

    # print(len(train_dataset))
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
        ),
        "val": DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    model = CNN_3D(num_classes=num_classes)  # CNN_timm(num_classes=num_classes)
    # print(model)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

    # swa_model = AveragedModel(model)

    if args.emd_loss:
        criterion = EMD()
    else:
        criterion = LabelSmoothingCrossEntropy()  # nn.CrossEntropyLoss(weight=weight)
    criterion_emd = EMD()

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.0
    )  # optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)#
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=args.decay_rate
    )  # CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    swa_start = 10
    # swa_scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.001, cycle_momentum=False) #SWALR(optimizer, swa_lr=0.0007)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train and validate

    scaler = GradScaler()

    for epoch in range(args.num_epochs):

        # Each epoch has a training and validation phase
        for split in ["train", "val"]:
            if split == "train":
                # wandb.watch(model, log_freq=10, log=all)
                model.train()  # Set model to training mode
                dataset_length = len(train_dataset)
            else:
                # do validation after certain number of epochs
                if epoch < 20:
                    break
                model.eval()  # Set model to evaluate mod
                dataset_length = len(validation_dataset)

            running_loss = 0.0
            running_corrects = 0
            y_true = []
            y_pred = []

            # Iterate over data.
            with tqdm(total=dataset_length) as epoch_pbar:
                epoch_pbar.set_description(f"Epoch {epoch}/{args.num_epochs - 1}")

                for batch, (image, target, frame_num, path) in enumerate(
                    dataloaders[split]
                ):
                    frame_num = frame_num.to(device)
                    inputs = image.to(device)
                    labels = target.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(split == "train"):
                        with autocast():
                            outputs = model(inputs, frame_num)[0]
                            # if epoch > swa_start and split == 'val':
                            #     swa_outputs = swa_model(inputs, frame_num)[0]
                            #     _, preds = torch.max(swa_outputs, 1)
                            # else:
                            _, preds = torch.max(outputs, -1)
                            loss = criterion(outputs, labels.long())
                            if args.emd_reg:
                                loss += args.loss_ratio * criterion_emd(
                                    outputs, labels.long()
                                )

                        if split == "train":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            # loss.backward()
                            # optimizer.step()
                            # prof.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    y_true = np.concatenate((labels.cpu().data.numpy(), y_true))
                    y_pred = np.concatenate((preds.cpu().data.numpy(), y_pred))

                    desc = (
                        f"Epoch {epoch}/{args.num_epochs - 1} - loss {loss.item():.4f}"
                    )
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(inputs.shape[0])

                    if split == "train":
                        if batch % 10 == 9:
                            wandb.log({f"{split} loss": loss})

            if split == "train":
                # if epoch > swa_start:
                #     swa_model.update_parameters(model)
                #     swa_scheduler.step()
                # else:
                scale = scaler.get_scale()
                skip_lr_sched = scale != scaler.get_scale()
                if not skip_lr_sched:
                    scheduler.step()
                # scheduler.step()

            epoch_loss = running_loss / dataset_length
            epoch_acc = accuracy_score(y_pred=y_pred, y_true=y_true)
            epoch_f1 = f1_score(y_pred=y_pred, y_true=y_true, average="macro")

            print(
                "{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}".format(
                    split, epoch_loss, epoch_acc, epoch_f1
                )
            )

            if split == "val":
                wandb.log({f"{split} f1_macro": epoch_f1})
                wandb.log({f"{split} accuracy": epoch_acc})

            # deep copy the model
            if split == "val":
                if epoch_f1 > best_acc:
                    best_acc = epoch_f1
                    epochs_no_improve = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_saved_path)
                else:
                    epochs_no_improve += 1

            # Early stopping
            if epoch > swa_start and epochs_no_improve == num_epochs_stop:
                print("Early stopping!")
                early_stop = True
                break

        if early_stop:
            break
    # torch.optim.swa_utils.update_bn(dataloaders['train'], swa_model(frame_num=frame_num))

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_saved_path)

    # test the model
    # y_pred = []
    # y_true = []
    # with torch.no_grad():
    #     for batch, (image, target, target1, target2) in enumerate(dataloaders['test']):
    #         inputs = image.to(device)
    #         labels = target2.to(device)
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #
    #         y_true = np.concatenate((labels.cpu().data.numpy(), y_true))
    #         y_pred = np.concatenate((preds.cpu().data.numpy(), y_pred))
    #
    # print(f1_score(y_pred=y_pred, y_true=y_true, average='macro'))


if __name__ == "__main__":
    run()
