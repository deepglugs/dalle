**DALL-E trainer and generator**

**Setup**

- Install DALLE-pytorch
```python
python3 -m pip install dalle-pytorch
```

**Training**

- Train VQVAE:

```bash
python3 dalle.py --source path/to/images/ \
				 --vae vae.pt \
				 --train_vae \
				 --batch_size=16 \
				 --samples_out samples/vae/ \
				 --epochs=2
```

- Train DALL-E

`--source` is expected to contain image files and tag files with the same name (ie, foo.png, foo.txt).

Tag files are a single-line file with comma separated tags.  Example:
```
1girl, white_swimsuit, red_hair
```

```bash
python3 dalle.py --source path/to/images/and/tags/ \
				 --vocab curated_512.vocab \
				 --vae vae.pt \
				 --train_dalle \
				 --dalle dalle.pt \
				 --batch_size=16 \
				 --samples_out samples/dalle/ \
				 --epochs=2
```

**Generating Images**

Generate a single image from tag file:

```bash
python3 dalle.py --generate --vocab curated_512.vocab --vae vae.pt --dalle dalle.pt --source my_tag.txt --output results/my_image.png
```

Generate images from a directory containing tag files:

```bash
python3 dalle.py --generate --vocab curated_512.vocab --vae vae.pt --dalle dalle.pt --source my/tags/ --output results
```

Generate an image using specified tags:

```bash
python3 dalle.py --generate --vocab curated_512.vocab --vae vae.pt --dalle dalle.pt --tags="1girl, white_swimsuit, red_hair" --output samples/my_image.png
```

