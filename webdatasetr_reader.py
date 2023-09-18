import webdataset as wds
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import io



transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

# dataset = wds.WebDataset('dataset.tar.gz').shuffle(8).decode().to_tuple("cap", "img")
dataset = wds.WebDataset('C:/DalleTrial/{0..9}.tar').decode()

# for d in dataset:
#     input(d)

# for val in dataset:
#     for item in val:
#         print(item)
#         input()

# dataloader = torch.utils.data.DataLoader(dataset)

# for val in dataloader:
#     input(val)

image_files = {d['__key__']: d['jpg'] for d in dataset}
text_files = {d['__key__']: d['txt'] for d in dataset}

keys = list(image_files.keys() & text_files.keys())

print(keys)
# print(type(image_files(key)))
i = 1
#
print(image_files)

# for key in keys:
#     print(type(img))
#     print(f"{i}) This is text for image above")
#     print(type(text_files[key]))
#     i = i+1

# for inp in dataloader:
#     input(inp)
    # for key in inp:
    #     print(inp[key])
    #     input()

# string = "This is a sample string"
# tokens = string.split()
# print(tokens)
