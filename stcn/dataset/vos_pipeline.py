from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset.range_transform import im_normalization, im_mean

# These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
pair_im_lone_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
])

pair_im_dual_transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
    transforms.Resize(384, InterpolationMode.BICUBIC),
    transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
])

pair_gt_dual_transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=0),
    transforms.Resize(384, InterpolationMode.NEAREST),
    transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
])

# These transform are the same for all pairs in the sampled sequence
all_im_lone_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
    transforms.RandomGrayscale(0.05),
])

all_im_dual_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=im_mean),
    transforms.RandomHorizontalFlip(),
])

all_gt_dual_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
    transforms.RandomHorizontalFlip(),
])

# Final transform without randomness
final_im_transform = transforms.Compose([
    transforms.ToTensor(),
    im_normalization,
])

final_gt_transform = transforms.Compose([
    transforms.ToTensor(),
])
