import os
import time

from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder

from torch.cuda.amp import autocast, GradScaler

from dalle_pytorch import DiscreteVAE, DALLE

from gan_utils import get_images, get_vocab, tokenize, is_image
from data_generator import ImageLabelDataset
from vgg_loss import VGGLoss

def loss_fn(x, y):
    return F.mse_loss(x, y) + F.smooth_l1_loss(x, y)


def get_vae(args):
    vae = DiscreteVAE(
        image_size=args.size,
        num_layers=args.vae_layers,
        num_tokens=8192,
        codebook_dim=args.codebook_dims,
        num_resnet_blocks=9,
        hidden_dim=128,
        temperature=args.temperature
    )

    if args.vae is not None and os.path.isfile(args.vae):
        print(f"loading state dict from {args.vae}")
        vae.load_state_dict(torch.load(args.vae))

    vae.to(args.device)

    return vae


def get_dalle(vae, vocab, args):
    dalle = DALLE(
            dim=args.codebook_dims,
            vae=vae,
            num_text_tokens=len(vocab)+1,
            text_seq_len=len(vocab),
            depth=16,
            heads=8,
            dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1,
            reversible=True
        )

    if args.dalle is not None and os.path.isfile(args.dalle):
        print(f"loading state dict from {args.dalle}")
        dalle.load_state_dict(torch.load(args.dalle))

    dalle.to(args.device)
    vae.to(args.device)

    return dalle


def train_vae(path, vocab, args):

    vae = get_vae(args)

    tforms = transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation((-180, 180)),
            # transforms.ColorJitter(0.15, 0.15, 0.15, 0.15),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)])

    #generator = DataGenerator(images,
    #                          txts,
    #                          vocab,
    #                          channels_first=True,
    #                          batch_size=1,
    #                         dim=(args.size, args.size),
    #                          normalize=False,
    #                          transform=tforms)

    generator = ImageFolder(args.source,
                            tforms)

    dl = torch.utils.data.DataLoader(generator,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=8)

    optim = Adam(vae.parameters(), lr=args.lr)

    vgg_loss = None

    if args.vgg_loss:
        vgg_loss = VGGLoss(device=args.device)

    rate = deque([1], maxlen=5)
    disp_size = 4
    step = 0

    if args.tempsched:
        vae.temperature = args.temperature
        dk = 0.7 ** (1 / len(generator))
        print('Scale Factor:', dk)

    for epoch in range(1, args.epochs):
        for i, data in enumerate(dl):
            step += 1

            t1 = time.monotonic()

            images, labels = data

            images = images.to(args.device)
            labels = labels.to(args.device)

            recons = vae(images)
            loss = loss_fn(images, recons)

            if vgg_loss is not None:
                loss += vgg_loss(images, recons)

            optim.zero_grad()

            loss.backward()

            optim.step()

            t2 = time.monotonic()
            rate.append(round(1.0 / (t2 - t1), 2))

            if step % 100 == 0:
                print("epoch {}/{} step {} loss: {} - {}it/s".format(epoch,
                                                   args.epochs,
                                                   step,
                                                   round(loss.item() / len(images), 6),
                                                   round(np.mean(rate)), 1))

            if step % 1000 == 0:
                with torch.no_grad():
                    codes = vae.get_codebook_indices(images[:disp_size])
                    imgx = vae.decode(codes)

                grid = torch.cat([images[:disp_size], recons[:disp_size], imgx])
                grid = make_grid(grid, nrow=disp_size, normalize=True, range=(-1, 1))
                VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"vae_{epoch}_{int(step / epoch)}.png"))
                print("saving checkpoint...")
                torch.save(vae.cpu().state_dict(), args.vae)
                vae.to(args.device)
                print("saving complete")

        if args.tempsched:
            vae.temperature *= dk
            print("Current temperature: ", vae.temperature)

    torch.save(vae.cpu().state_dict(), args.vae)


def train_dalle(vae, args):

    if args.vocab is None:
        args.vocab = args.source
    else:
        assert os.path.isfile(args.vocab)

    if args.tags_source is None:
        args.tags_source = args.source

    imgs = get_images(args.source)
    txts = get_images(args.tags_source, exts=".txt")
    vocab = get_vocab(args.vocab, top=args.vocab_limit)

    tforms = transforms.Compose([
             transforms.Resize((args.size, args.size)),
             transforms.ToTensor(),
             transforms.Normalize((0.5,)*3, (0.5,)*3)])

    def txt_xforms(txt):
        # print(f"txt: {txt}")
        txt = txt.split(", ")
        if args.shuffle_tags:
            np.random.shuffle(txt)
        txt = tokenize(txt, vocab, offset=1)
        # txt = torch.Tensor(txt)

        return txt

    data = ImageLabelDataset(imgs, txts, vocab,
                        dim=(args.size, args.size),
                        transform=tforms,
                        channels_first=True,
                        return_raw_txt=True)

    dl = torch.utils.data.DataLoader(data,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=0)

    dalle = get_dalle(vae, vocab, args)

    optimizer = Adam(dalle.parameters(), lr=args.lr)
    if args.vgg_loss:
        vgg_loss = VGGLoss(device=args.device)

    disp_size = 4 if args.batch_size > 4 else args.batch_size

    amp_scaler = GradScaler(enabled=args.fp16)

    for epoch in range(1, args.epochs + 1):

        batch_idx = 0
        train_loss = 0

        for image, labels in dl:
            i = image
            text_ = []
            for label in labels:
                text_.append(txt_xforms(label))
            # print(text_)
            text = torch.LongTensor(text_).to(args.device)
            image = image.to(args.device)

            mask = torch.ones_like(text).bool().to(args.device)

            with autocast(enabled=args.fp16):
                loss = dalle(text, image, mask=mask, return_loss=True)

            # loss = loss_func(image, gens)
            train_loss += loss.item()

            optimizer.zero_grad()
            amp_scaler.scale(loss).backward()

            amp_scaler.step(optimizer)
            amp_scaler.update()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(i), len(data),
                    100. * batch_idx / int(round(len(data) / args.batch_size)),
                    loss.item() / len(image)))

            if batch_idx % 100 == 0:
                oimgs = dalle.generate_images(text, mask=mask)
                grid = oimgs[:disp_size]
                grid = make_grid(grid, nrow=disp_size,
                                 normalize=True, range=(-1, 1))
                VTF.to_pil_image(grid).save(
                    os.path.join(args.samples_out,
                                 f"dalle_{epoch}_{int(batch_idx)}.png"))
                torch.save(dalle.cpu().state_dict(), args.dalle)
                dalle.to(args.device)

            batch_idx += 1

        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data)))

    torch.save(dalle.cpu().state_dict(), args.dalle)
    dalle.to(args.device)


def generate(args):

    vae = get_vae(args)

    assert os.path.isfile(args.vocab)

    if args.tags_source is None:
        args.tags_source = args.source

    if args.tags_source is not None:
        txts = get_images(args.tags_source, exts=".txt")

    if args.tags_source is not None and os.path.isfile(args.tags_source):
        txts = [args.tags_source]

    vocab = get_vocab(args.vocab, top=args.vocab_limit)

    dalle = get_dalle(vae, vocab, args)

    def txt_xforms(txt):
        txt = txt.split(", ")
        txt = tokenize(txt, vocab)
        txt = torch.Tensor(txt)

        return txt

    tags = []

    if args.tags is not None:
        out_fn = os.path.basename(args.output)

        if not is_image(args.output):
            out_fn = "_".join(args.tags.split(", "))
        else:
            args.output = os.path.dirname(args.output)

        tags = [(txt_xforms(args.tags), out_fn)]

    else:
        # preload tags       
        for txt in txts:
            with open(txt) as f:
                txt = os.path.splitext(os.path.basename(txt))[0]

                if is_image(args.output):
                    txt = os.path.basename(args.output)
                    args.output = os.path.dirname(args.output)

                tags.append((txt_xforms(f.read()), txt))

    for tag, out_fn in tags:
        tag = torch.LongTensor(tag.long()).to(args.device)
        mask = torch.ones_like(tag).bool().to(args.device)

        oimg = dalle.generate_images(tag.view((1, -1)), mask=mask.view((1, -1)))[0]

        output = os.path.join(args.output, out_fn)
        os.makedirs(args.output, exist_ok=True)

        if not is_image(output):
            # probably doesn't have an extension yet?
            output += ".webp"

        print(f"output to: {output}")

        save_image(oimg, output, normalize=True)

        # VTF.to_pil_image(oimg).save(
        #    os.path.join(args.output, out_fn) + ".webp",
        #    "WEBP",
        #    lossless=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--tags_source', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--vae', default=None)
    parser.add_argument('--dalle', default=None)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--size', default=128, type=int)
    parser.add_argument('--vocab_limit', default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--samples_out', default="samples")
    parser.add_argument('--train_vae', action='store_true')
    parser.add_argument('--train_dalle', action='store_true')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--tempsched', action='store_true')
    parser.add_argument('--shuffle_tags', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--vgg_loss', action='store_true')
    parser.add_argument('--vae_layers', default=2, type=int)
    parser.add_argument('--codebook_dims', default=1024, type=int)
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='vae temperature (default: 0.9)')

    args = parser.parse_args()

    os.makedirs(args.samples_out, exist_ok=True)

    if args.train_vae:
        vae = train_vae(args.source,
                        args.vocab,
                        args)

    if args.train_dalle:
        vae = get_vae(args)
        train_dalle(vae, args)

    if args.generate:
        generate(args)


if __name__ == "__main__":
    main()
