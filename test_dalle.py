import os

from dalle import generate


def base_setup():
    class Args(object):
        pass

    args = Args()

    args.generate = True
    args.vae_layers = 2
    args.vocab = "curated_512.vocab"
    args.vae = "vae_ch_3.pt"
    args.dalle = "dalle_ch.pt"
    args.codebook_dims = 1024
    args.size = 128
    args.temperature = 0.9
    args.device = "cuda"
    args.vocab_limit = None
    args.tags = None
    args.tags_source = None
    args.source = None

    return args


def test_generate_tag_file():

    args = base_setup()

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:

        tag_file = os.path.join(tmpdir, "test_tags.txt")

        with open(tag_file, 'w') as f:
            f.write("1girl, white_swimsuit")

        args.tags_source = tag_file
        args.output = tmpdir

        generate(args)

        out_fn = os.path.join(tmpdir, 'test_tags.webp')

        assert os.path.isfile(out_fn)

def test_generate_tag_dir():

    args = base_setup()

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:

        tag_file = os.path.join(tmpdir, "test_tags.txt")

        with open(tag_file, 'w') as f:
            f.write("1girl, white_swimsuit")

        args.tags_source = tmpdir
        args.output = tmpdir

        generate(args)

        out_fn = os.path.join(tmpdir, 'test_tags.webp')

        assert os.path.isfile(out_fn)

def test_generate_using_source():

    args = base_setup()

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:

        tag_file = os.path.join(tmpdir, "test_tags.txt")

        with open(tag_file, 'w') as f:
            f.write("1girl, white_swimsuit")

        args.source = tmpdir
        args.output = tmpdir

        generate(args)

        out_fn = os.path.join(tmpdir, 'test_tags.webp')

        assert os.path.isfile(out_fn)

def test_generate_tags_dir_out():

    args = base_setup()

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:

        args.tags = "1girl, white_swimsuit"
        args.output = tmpdir

        generate(args)

        out_fn = os.path.join(tmpdir, '1girl_white_swimsuit.webp')

        assert os.path.isfile(out_fn)

def test_generate_tags_set_file_out():

    args = base_setup()

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:

        out_fn = os.path.join(tmpdir, "my_cool_gen.webp")

        args.tags = "1girl, white_swimsuit"
        args.output = out_fn

        generate(args)

        assert os.path.isfile(out_fn)
