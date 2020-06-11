import os
from Gan import Gan
import numpy as np


def createVideo(contrast, num_cycles):
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

        old_seed = new_seed

    os.system("ffmpeg -y -framerate 25 -i ../generated_video/frames/test_%03d.png ../generated_video/latentSpaceNavigation.mov")


def interpolate_points(p1, p2, n_steps=100):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


createVideo(2, 1)
