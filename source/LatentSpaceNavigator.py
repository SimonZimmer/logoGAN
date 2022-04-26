import os
from Gan import Gan
import numpy as np
import sys
import PIL.Image as Image
sys.path.append("../ThirdParty/super-resolution")
from model.srgan import generator
from model import resolve_single


def createVideo(contrast, num_cycles, sr_model):
    video_dir = "../generated_video"
    frames_dir = os.path.join(video_dir, "frames")
    if not os.path.isdir(video_dir):
        os.mkdir(video_dir)
    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)
    model = Gan(train=False)
    model.loadGenerator('../deploy_models/generator_at_epoch300.h5')

    old_seed = np.random.uniform(-contrast, contrast, (1, 100))
    for cycle in range(num_cycles):
        new_seed = np.random.uniform(-contrast, contrast, (1, 100))
        interpolated = interpolate_points(old_seed, new_seed)

        for vector in range(interpolated.shape[0]):
            primer = interpolated[vector]
            image = model.generate_image(primer)
            image_path = os.path.join(video_dir, "frames", f"test_{str((cycle * 100) + vector).zfill(3)}.png")
            print(f"saving image for cycle {cycle} and vector {vector} in {image_path}")
            model.save_image(image, image_path)

            upscale_image(image_path, sr_model)

        old_seed = new_seed

    os.system(f"ffmpeg -y -framerate 25 -i ../generated_video/frames/test_%03d.png ../generated_video/latentSpaceNavigation_{contrast}.mov")


def interpolate_points(p1, p2, n_steps=100):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


def upscale_image(filePath, model):
    image = np.asarray(Image.open(filePath).convert('RGB'))
    upscaled = resolve_single(model, image)
    upscaled_resized = Image.fromarray(np.asarray(upscaled)).resize((256, 256))
    upscaled_resized = np.asarray(upscaled_resized)
    upscaled_second = resolve_single(model, upscaled_resized)
    image_processed = Image.fromarray(np.asarray(upscaled_second)).convert('L')
    image_processed.save(filePath)


def generate_random_image_set(num_images):
    random_image_dir = "../random_images"
    if not os.path.isdir(random_image_dir):
        os.mkdir(random_image_dir)
    model = Gan(train=False)
    model.loadGenerator('../deploy_models/generator_at_epoch300.h5')

    for image_index in range(num_images):
        primer = np.random.uniform(-2, 2, (1, 100))
        image = model.generate_image(primer)
        primerText = 0
        for num in range(primer.shape[1]):
            primerText += primer[0, num]
        image_path = os.path.join(random_image_dir, f"random_image_{str(primerText)}.png")
        model.save_image(image, image_path)
        upscale_image(image_path, sr_model)


sr_model = generator()
sr_model.load_weights('../ThirdParty/super-resolution/weights/srgan/gan_generator.h5')
createVideo(1.8, 200, sr_model)

