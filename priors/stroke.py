from PIL import Image, ImageDraw, ImageFilter
import random
import math

import torch
import numpy as np
from .utils import get_batch_to_dataloader

def mnist_prior(num_classes=2, size=28, min_max_strokes=(1,3), min_max_len=(5/28,20/28), min_max_start=(2/28,25/28),
                min_max_width=(1/28,4/28), max_offset=4/28, max_target_offset=2/28):
    classes = []
    for i in range(num_classes):
        num_strokes = random.randint(*min_max_strokes)
        len_strokes = [random.randint(int(size * min_max_len[0]), int(size * min_max_len[1])) for i in range(num_strokes)]
        stroke_start_points = [
            (random.randint(int(size * min_max_start[0]), int(size * min_max_start[1])), random.randint(int(size * min_max_start[0]), int(size * min_max_start[1]))) for i in
            range(num_strokes)]
        stroke_directions = []
        # i = Image.fromarray(np.zeros((28,28),dtype=np.uint8))
        # draw = ImageDraw.Draw(i)
        for i in range(num_strokes):
            sp, length = stroke_start_points[i], len_strokes[i]
            counter = 0
            while True:
                if counter % 3 == 0:
                    length = random.randint(int(size * min_max_len[0]), int(size * min_max_len[1]))
                    sp = (
                    random.randint(int(size * min_max_start[0]), int(size * min_max_start[1])), random.randint(int(size * min_max_start[0]), int(size * min_max_start[1])))
                    stroke_start_points[i], len_strokes[i] = sp, length
                radians = random.random() * 2 * math.pi
                x_vel = math.cos(radians) * length
                y_vel = math.sin(radians) * length
                new_p = (sp[0] + x_vel, sp[1] + y_vel)
                # print(math.degrees(radians),sp,new_p)
                if not any(n > size - 1 or n < 0 for n in new_p):
                    break
                counter += 1
            stroke_directions.append(radians)
            # print([round(x) for x in sp+new_p])
            # draw.line([round(x) for x in sp+new_p], fill=128, width=3)
        classes.append((len_strokes, stroke_start_points, stroke_directions))

    generator_functions = []
    for c in classes:
        def g(c=c):
            len_strokes, stroke_start_points, stroke_directions = c
            i = Image.fromarray(np.zeros((size, size), dtype=np.uint8))
            draw = ImageDraw.Draw(i)
            width = random.randint(int(size * min_max_width[0]), int(size * min_max_width[1]))
            offset = random.randint(int(-size * max_offset), int(size * max_offset)), random.randint(int(- size * max_offset), int(size * max_offset))
            for sp, length, radians in zip(stroke_start_points, len_strokes, stroke_directions):
                sp = (sp[0] + offset[0], sp[1] + offset[1])
                x_vel = math.cos(radians) * length + random.randint(int(-size * max_target_offset), int(size * max_target_offset))
                y_vel = math.sin(radians) * length + random.randint(int(-size * max_target_offset), int(size * max_target_offset))
                new_p = (sp[0] + x_vel, sp[1] + y_vel)
                stroke_directions.append(radians)
                draw.line([round(x) for x in sp + new_p], fill=128, width=width)
            a_i = np.array(i)
            a_i[a_i == 128] = np.random.randint(200, 255, size=a_i.shape)[a_i == 128]
            return Image.fromarray(a_i).filter(ImageFilter.GaussianBlur(.2))

        generator_functions.append(g)
    return generator_functions


# g1,g2 = mnist_prior(2)

# for i in [g1() for _ in range(10)]:
#    display(i.resize((200,200)))

from torchvision.transforms import ToTensor, ToPILImage


def normalize(x):
    return (x-x.mean())/(x.std()+.000001)

from os import path, listdir
import random

def get_batch(batch_size, seq_len, num_features=None, noisy_std=None, only_train_for_last_idx=False, normalize_x=False, num_outputs=2, use_saved_from=None, **kwargs):  # num_features = 28*28=784
    if use_saved_from is not None:
        directory = path.join(use_saved_from, f'len_{seq_len}_out_{num_outputs}_features_{num_features}_bs_{batch_size}')
        filename = random.choice(listdir(directory))
        return torch.load(path.join(directory,filename))

    size = math.isqrt(num_features)
    assert size * size == num_features, 'num_features needs to be the square of an integer.'
    if only_train_for_last_idx:
        assert (seq_len-1) % num_outputs == 0

    # assert seq_len % 2 == 0, "assert seq_len % 2 == 0"
    batch = []
    y = []
    target_y = []
    for b_i in range(batch_size):
        gs = mnist_prior(num_outputs, size, **kwargs)
        if only_train_for_last_idx:
            generators = [i for i in range(len(gs)) for _ in range((seq_len-1) // num_outputs)]
            random.shuffle(generators)
            generators += [random.randint(0, len(gs) - 1)]
            target = [-100 for _ in generators]
            target[-1] = generators[-1]
        else:
            generators = [random.randint(0, len(gs) - 1) for _ in range(seq_len)]
            target = generators
        normalize_or_not = lambda x: normalize(x) if normalize_x else x
        s = torch.cat([normalize_or_not(ToTensor()(gs[f_i]())) for f_i in generators], 0)
        batch.append(s)
        y.append(torch.tensor(generators))
        target_y.append(torch.tensor(target))
    x = torch.stack(batch, 1).view(seq_len, batch_size, -1)
    y = torch.stack(y, 1)
    target_y = torch.stack(target_y, 1)
    return x,y,target_y

DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 2

if __name__ == '__main__':
    g1, g2 = mnist_prior(2, size=3)

    # for i in range(10):
    # print(PILToTensor()(g1()))
    # display(ToPILImage()(PILToTensor()(g1())).resize((200,200)))
    # display(g2().resize((200,200)))

    size = 10
    x, y = get_batch(1, 10, num_features=size * size)

    x_ = x[..., :-1].squeeze(1)
    last_y = x[..., -1].squeeze(1)
    y = y.squeeze(1)

    # print(y)

    for i, y_, last_y_, x__ in zip(x_, y, last_y, x.squeeze(1)):
        # print(y_)
        # print(i.shape)
        # print(x__)
        img = ToPILImage()(i.view(size, size))
        # display(img.resize((200,200)))

    print(y, last_y)
