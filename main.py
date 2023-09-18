import torch
from dalle2_pytorch import Unet, DALLE2, Decoder, DiffusionPriorNetwork, DiffusionPrior, CLIP
from torchvision.utils import save_image
from dalle2_pytorch.dataloaders import ImageEmbeddingDataset, create_image_embedding_dataloader
from dalle2_pytorch.dataloaders import get_reader, make_splits
from torchvision import transforms
import torchvision.transforms.functional as fn
import gc
import sys

# del variables
# gc.collect()
# image = read_image(str(self.tile_filenames[idx]))
#         assert len(image.shape) == 3 and tuple(image.shape[1:]) == (256, 256)
# Create a dataloader directly.
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])
for tar in range(10):
    print("This is the beginning of the loop")
    dataloader = create_image_embedding_dataloader(
        tar_url=f"C:/DalleTrial/{tar}.tar",  # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
        # tar_url="C:/DalleTrial/0.tar",  # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
        # embeddings_url="path/or/url/to/embeddings/folder",     # Included if .npy files are not in webdataset. Left out or set to None otherwise
        num_workers=4,
        batch_size=32,
        # shard_width=4,                                         # If a file in the webdataset shard 3 is named 0003039.jpg, we know the shard width is 4 and the last three digits are the index
        shuffle_num=200,                                       # Does a shuffle of the data with a buffer size of 200
        shuffle_shards=True,                                   # Shuffle the order the shards are read in
        resample_shards=False,                                 # Sample shards with replacement. If true, an epoch will be infinite unless stopped manually
    )
    # C:\Users\Tirth>pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
    # dataloader = create_image_embedding_dataloader(
    #     # tar_url="C:/DalleTrial/{0..9}.tar",  # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
    #     # tar_url="C:/DalleTrial/0.tar",  # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
    #     # embeddings_url="path/or/url/to/embeddings/folder",     # Included if .npy files are not in webdataset. Left out or set to None otherwise
    #     img_embeddings_url=None,
    #     text_embeddings_url=None,
    #     num_workers=4,
    #     batch_size=32,
    #     # shard_width=4,                                         # If a file in the webdataset shard 3 is named 0003039.jpg, we know the shard width is 4 and the last three digits are the index
    #     shuffle_num=200,                                       # Does a shuffle of the data with a buffer size of 200
    #     shuffle_shards=True,                                   # Shuffle the order the shards are read in
    #     resample_shards=False,                                 # Sample shards with replacement. If true, an epoch will be infinite unless stopped manually
    # )

    # print("I am here")
    # ex1 = torch.randn(3, 256, 256)
    # print(ex1.shape)
    # ex2 = torch.randn(3, 450, 300)
    # print(ex2.shape)
    # ex3 = torch.stack((ex1, ex2))
    # print(ex3.shape)


    # text = torch.randint(0, 49408, (4, 256)).cuda()
    list_of_image = []
    list_of_text = []
    for img, emb in dataloader.dataset:
        # img = fn.center_crop(img, output_size=[100])
        img = fn.resize(img, size=[256, 256])
        list_of_image.append(transform(img))
        # print(transform(img).shape)
        text = str(emb)
        text = text.replace(f"\n", "")
        list_of_text.append(text)
        # print(transform(img).shape)
        # print(emb)
        # print(transform(img).shape)  # torch.Size([32, 3, 256, 256])
        # print(txt)
        # print(text.shape)# torch.Size([32, 512])
        # Train decoder only as shown above

    # print(list_of_text)
    # tokens = ['cat'-69, 'motel'-67, 'small cat', 'cat' - 69, 'string']
    word_to_idx = {word: i for i, word in enumerate(list_of_text)}
    # print(word_to_idx)
    # numerical_values = [word_to_idx[word] for word in list_of_text]
    # text_tensor = torch.tensor(numerical_values)

    list_of_text_tensor = []  # [tensor(69), tensor(67), ......]
    i = 0
    for word in list_of_text:
        list_of_text_tensor.append(torch.tensor((i, word_to_idx[word])))
        i = i + 1

    # print(list_of_text_tensor)
    # print(list_of_image)

    text_tensor = torch.stack(list_of_text_tensor) # tensor.size(70, 2)
    image_tensor = torch.stack(list_of_image)
    # print(sys.getsizeof(image_tensor))
    # print(image_tensor.element_size())
    # print(image_tensor.shape)
    # print("done!")
    # print(text_tensor.shape)
    # print(image_tensor.shape)
    # print("I am here part 2")
    # del list_of_text_tensor, word_to_idx, list_of_text, list_of_image, text, word,
    # gc.collect()

    clip = CLIP(
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 49408,
        text_enc_depth = 1,
        text_seq_len = 256,
        text_heads = 8,
        visual_enc_depth = 1,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_heads = 8,
        use_all_token_embeds = True,            # whether to use fine-grained contrastive learning (FILIP)
        decoupled_contrastive_learning = True,  # use decoupled contrastive learning (DCL) objective function, removing positive pairs from the denominator of the InfoNCE loss (CLOOB + DCL)
        extra_latent_projection = True,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_visual_ssl = True,                  # whether to do self supervised learning on images
        visual_ssl_type = 'simclr',             # can be either 'simclr' or 'simsiam', depending on using DeCLIP or SLIP
        use_mlm = False,                        # use masked language learning (MLM) on text (DeCLIP)
        text_ssl_loss_weight = 0.05,            # weight for text MLM loss
        image_ssl_loss_weight = 0.05            # weight for image self-supervised learning loss
    ).cuda()

    # mock data

    # train [Added a loop her - Tirth Patel]
    text = text_tensor.cuda()
    images = image_tensor.cuda()

    loss = clip(
        text,
        images,
        return_loss = True              # needs to be set to True to return contrastive loss
    ).cuda()

    loss.backward()

    # text = torch.randint(0, 49408, (4, 256)).cuda()
    # images = torch.randn(4, 3, 256, 256).cuda()
    #
    # # train
    #
    # loss = clip(
    #     text,
    #     images,
    #     return_loss = True              # needs to be set to True to return contrastive loss
    # )
    #
    # loss.backward()
    # do the above with as many texts and images as possible in a loop

    # trained clip from step 1

    clip = CLIP(
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 49408,
        text_enc_depth = 1,
        text_seq_len = 256,
        text_heads = 8,
        visual_enc_depth = 1,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_heads = 8
    ).cuda()

    # unet for the decoder

    unet = Unet(
        dim = 128,
        image_embed_dim = 512,
        cond_dim = 128,
        channels = 3,
        dim_mults=(1, 2, 4, 8)
    ).cuda()

    # decoder, which contains the unet and clip
    torch.cuda.empty_cache()
    decoder = Decoder(
        unet = unet,
        clip = clip,
        timesteps = 100,
        image_cond_drop_prob = 0.1,
        text_cond_drop_prob = 0.5
    ).cuda()

    # mock images (get a lot of this)
    images = image_tensor.cuda()

    # feed images into decoder
    torch.cuda.empty_cache()
    loss = decoder(images).cuda()
    loss.backward()
    torch.cuda.empty_cache()
    # do the above for many many many many steps
    # then it will learn to generate images based on the CLIP image embeddings

    # get trained CLIP from step one

    clip = CLIP(
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 49408,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        visual_enc_depth = 6,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_heads = 8,
    ).cuda()

    # setup prior network, which contains an autoregressive transformer

    prior_network = DiffusionPriorNetwork(
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8
    ).cuda()

    # diffusion prior network, which contains the CLIP and network (with transformer) above

    diffusion_prior = DiffusionPrior(
        net = prior_network,
        clip = clip,
        timesteps = 100,
        cond_drop_prob = 0.2
    ).cuda()

    # mock data
    text = text_tensor.cuda()
    images = image_tensor.cuda()

    # feed text and images into diffusion prior network

    loss = diffusion_prior(text, images).cuda()
    loss.backward()

    # do the above for many many many steps
    # now the diffusion prior can generate image embeddings from the text embeddings

    # trained clip from step 1

    clip = CLIP(
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 49408,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        visual_enc_depth = 6,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_heads = 8
    ).cuda()

    # 2 unets for the decoder (a la cascading DDPM)

    unet1 = Unet(
        dim = 32,
        image_embed_dim = 512,
        cond_dim = 128,
        channels = 3,
        dim_mults = (1, 2, 4, 8)
    ).cuda()

    unet2 = Unet(
        dim = 32,
        image_embed_dim = 512,
        cond_dim = 128,
        channels = 3,
        dim_mults = (1, 2, 4, 8, 16)
    ).cuda()

    # decoder, which contains the unet(s) and clip

    decoder = Decoder(
        clip = clip,
        unet = (unet1, unet2),            # insert both unets in order of low resolution to highest resolution (you can have as many stages as you want here)
        image_sizes = (128, 256),         # resolutions, 256 for first unet, 512 for second. these must be unique and in ascending order (matches with the unets passed in)
        timesteps = 1000,
        image_cond_drop_prob = 0.1,
        text_cond_drop_prob = 0.5
    ).cuda()


    # mock images (get a lot of this)
    images = image_tensor.cuda()

    # feed images into decoder, specifying which unet you want to train
    # each unet can be trained separately, which is one of the benefits of the cascading DDPM scheme

    loss = decoder(images, unet_number = 1).cuda()
    loss.backward()

    loss = decoder(images, unet_number = 2).cuda()
    loss.backward()
    print(f"We are at {tar} tar file")

    # do the above for many steps for both unets

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
).cuda()

# send the text as a string if you want to use the simple tokenizer from DALLE v1
# or you can do it as token ids, if you have your own tokenizer

texts = ['a cat']
images = dalle2(texts).cuda()  # (1, 3, 256, 256)

# saving the image

img1 = images[0]
save_image(img1, 'img1.png')
