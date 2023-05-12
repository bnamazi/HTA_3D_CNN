from torchvision import transforms, utils
import torch


transform_train = torch.nn.Sequential(
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandAugment()]), 0.85),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
    # transforms.RandomErasing(
    #     p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False
    # ),
)

transform_val = torch.nn.Sequential(
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
)

# transform_train = transforms.Compose(
#     [
#         transforms.Resize(
#             (300, 300), interpolation=transforms.InterpolationMode.BICUBIC
#         ),
#         transforms.RandomApply(
#             [
#                 transforms.RandomOrder(
#                     (
#                         # transforms.RandomResizedCrop(
#                         #     224,
#                         #     scale=(0.85, 1.20),
#                         #     ratio=(0.75, 1.3333333333333333),
#                         #     interpolation=transforms.InterpolationMode.BICUBIC,
#                         # ),
#                         # transforms.RandomAffine(
#                         #     degrees=10,
#                         #     translate=(0.05, 0.05),
#                         #     scale=None,
#                         #     shear=0.05,
#                         #     fill=0,
#                         # ),
#                         transforms.ColorJitter(
#                             brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
#                         ),
#                         transforms.RandomGrayscale(p=0.05),
#                         # transforms.RandomVerticalFlip(p=0.1),
#                         # transforms.RandomHorizontalFlip(p=0.5),
#                     )
#                 )
#             ],
#             0.85,
#         ),
#         # transforms.Resize(
#         #     (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
#         # ),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
#         transforms.RandomErasing(
#             p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False
#         ),
#     ]
# )

# transform_val = transforms.Compose(
#     [
#         transforms.Resize(
#             (300, 300), interpolation=transforms.InterpolationMode.BICUBIC
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
#     ]
# )
