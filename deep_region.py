import argparse
import os
from datetime import datetime, time

from PIL import Image

import data
import models
import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functions import percentage_block
from functions import mutation_vectors_block_l2
import random
from torchvision import transforms
# from grad_cam import*

def select_region(model, x, logits_clean, truth, regions, region_size):
    """为每个区域添加噪声并进行预测"""
    c, h, w = x.shape[1:]
    region_diffs = []  # 用于保存每个区域的预测差异
    all_mutated_images = []  # 用于保存所有变异后的区域

    for start_i, start_j, region in regions:
        # 为当前区域添加噪声
        target_norm = 0.007
        noise = np.random.uniform(-1, 1, size=region.shape)
        noise_norm = np.sqrt(np.sum(noise ** 2, axis=(1, 2, 3), keepdims=True)) + 1e-8
        scaled_noise = (noise / noise_norm) * target_norm
        mutated_region = np.clip(region + scaled_noise, 0, 1)

        # 将变异后的区域放回原图
        x_mutated = x.copy()
        x_mutated[:, :, start_i:min(start_i + region_size, h), start_j:min(start_j + region_size, w)] = mutated_region
        all_mutated_images.append(x_mutated)
        # l2_norm = np.sqrt(np.sum((x_mutated - x) ** 2, axis=(1, 2, 3), keepdims=True))

    # 批量预测所有扰动图像
    all_mutated_images = np.concatenate(all_mutated_images, axis=0)  # 将所有扰动图像合并
    all_logits = model.predict(all_mutated_images)
    query_selection = all_logits.shape[0]

    # 计算扰动对目标类别的影响
    for i, (start_i, start_j, region) in enumerate(regions):
        logits = all_logits[i]
        diff = abs(logits_clean[0][truth] - logits[truth])  # 计算差异
        region_diffs.append((start_i, start_j, region, diff))
    # print('finished predicting')

    # 选择区域
    num = int(len(regions) * 0.63)
    # num = len(regions) // 2
    top_regions = sorted(region_diffs, key=lambda x: x[3], reverse=True)[:num]
    selected_regions = [(start_i, start_j, region) for start_i, start_j, region, _ in top_regions]

    mask = np.zeros_like(x)
    for start_i, start_j, region in selected_regions:
        region_h, region_w = region.shape[2], region.shape[3]
        mask[:, :, start_i:start_i + region_h, start_j:start_j + region_w] = 1


    return selected_regions, query_selection, mask

def random_select_region(x, region_stop):
    """随机选择区域"""
    c, h, w = x.shape[1:]
    regions = []

    for i in range(0, h, region_stop):
        for j in range(0, w, region_stop):
            region = x[:, :, i:min(i + region_stop, h), j:min(j + region_stop, w)]
            regions.append((i, j, region))

    random.shuffle(regions)
    num_regions = max(1, len(regions) // 4)
    selected_regions = regions[:num_regions]

    mask = np.zeros_like(x)
    for start_i, start_j, region in selected_regions:
        region_h, region_w = region.shape[2], region.shape[3]
        mask[:, :, start_i:start_i + region_h, start_j:start_j + region_w] = 1

    return selected_regions, mask


def split_region(x, regions, region_size):
    """对区域进行对半分"""
    c, h, w = x.shape[1:]
    # print(regions)
    split_regions = []
    new_region_size = region_size // 2

    for index in regions:
        start_i = index[0]
        start_j = index[1]
        region = index[2]
        for i in range(start_i, min(start_i + region_size, h), new_region_size):
            for j in range(start_j, min(start_j + region_size, w), new_region_size):
                new_region = x[:, :, i:min(i + new_region_size, h), j:min(j + new_region_size, w)]
                split_regions.append((i, j, new_region))

    return split_regions, new_region_size

def visualize_selected_regions(image, selected_regions):
    # plt.imshow(image.transpose(1, 2, 0))
    if image.shape[0] == 1:
        plt.imshow(image[0], cmap='gray')
    else:
        plt.imshow(image.transpose(1, 2, 0))

    ax = plt.gca()

    for (start_i, start_j, region) in selected_regions:
        rect = patches.Rectangle((start_j, start_i), region.shape[3], region.shape[2],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

def mutation_vector_l2(start_i, start_j, region):
    c, h, w = x.shape[1:]
    # region_size = region.shape[2]
    mutation_block = np.zeros((1, 1, region.shape[2], region.shape[3]))


    x_c = np.random.randint(0, region.shape[2])
    y_c = np.random.randint(0,region.shape[3])

    # radius = region_size
    for counter_x in range(0, region.shape[2]):
        for counter_y in range(0, region.shape[3]):
            distance_power = (counter_x - x_c)**2 + (counter_y - y_c)**2
            if distance_power == 0.0:
                mutation_block[:, :, counter_x, counter_y] = 0.8
            else:
                mutation_block[:, :, counter_x, counter_y] = 0.4/distance_power

    mutation_block /= np.sqrt(np.sum(mutation_block ** 2, keepdims=True))

    return mutation_block

def perturb_region(model, x, y, objective_clean, selected_regions, truth, eps, n_iters):
    c, h, w = x.shape[1:]
    x_mutated = x.copy()
    succ = False
    query_perturb = 0
    dist = 0
    objective = objective_clean
    error_counts = np.zeros(1000, dtype=int)

    mutation = np.zeros_like(x)
    for start_i, start_j, region in selected_regions:
        # print(start_i, start_j)
        # print(region.shape)
        mutation_block = mutation_vector_l2(start_i, start_j, region)
        mutation[:, :, start_i:min(start_i + region.shape[2], h), start_j:min(start_j + region.shape[3], w)] = mutation_block * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])

    x_mutated = np.clip(x_mutated + mutation / np.sqrt(np.sum(mutation ** 2, keepdims=True)) * eps, 0, 1)

    # 模型预测
    logits_mutated = model.predict(x_mutated)
    objective_mutated = model.objective(y, logits_mutated)
    objective = objective_mutated
    query_perturb += 1

    # 检查是否预测错误
    if np.argmax(logits_mutated) != truth:
        # print("Attack successful!")
        succ = True
        dist = np.linalg.norm(x_mutated - x)
        save_image(x_mutated[0], truth, j, f"./results2/attack_success/{args.model}")
        # print("Distance:", dist)
        return succ, query_perturb, dist

    # perturb_start = datetime.now()
    for index in range(n_iters):
        mutation_index = x_mutated - x

        # mutation insert
        insertion_region = random.choice(selected_regions)
        inser_i, inser_j, inser_region = insertion_region
        new_mutation_block = mutation_vector_l2(inser_i, inser_j, inser_region)
        new_mutations = np.ones([x.shape[0], c, inser_region.shape[2], inser_region.shape[3]]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
        mutation_insertion_block = np.zeros(x.shape)
        mutation_insertion_block[:, :, inser_i:inser_i + inser_region.shape[2], inser_j:inser_j + inser_region.shape[3]] = 1

        # mutation delete
        mutation_deletion_blocks = np.zeros(x.shape)
        i = 0
        while i < 10:
            delete_region = random.choice(selected_regions)
            start_i, start_j, region = delete_region
            mutation_deletion_blocks[:, :, start_i:start_i + region.shape[2], start_j:start_j + region.shape[3]] = 1
            mutation_index[:, :, start_i:start_i + region.shape[2], start_j:start_j + region.shape[3]] = 0
            i += 1

        distance_insertion_block = np.sqrt(
            np.sum(((x_mutated - x) * mutation_insertion_block) ** 2, axis=(2, 3), keepdims=True))
        distance_mutated_image = np.sqrt(np.sum((x_mutated - x) ** 2, axis=(1, 2, 3), keepdims=True))
        all_blocks = np.maximum(mutation_insertion_block, mutation_deletion_blocks)
        distance_all_blocks = np.sqrt(np.sum((mutation_index * all_blocks) ** 2, axis=(2, 3), keepdims=True))

        old_mutations = mutation_index[:, :, inser_i:inser_i + inser_region.shape[2], inser_j:inser_j + inser_region.shape[3]] / (1e-10 + distance_insertion_block)
        new_mutations = old_mutations + new_mutations
        new_mutations = new_mutations / np.sqrt(np.sum(new_mutations ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum(eps ** 2 - distance_mutated_image ** 2, 0) / c + distance_all_blocks ** 2) ** 0.5
        mutation_index[:, :, inser_i:inser_i + inser_region.shape[2], inser_j:inser_j + inser_region.shape[3]] = new_mutations + 0

        x_new = x + mutation_index / np.sqrt(np.sum(mutation_index ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, 0, 1)

        logits_new = model.predict(x_new)
        predicted_label = np.argmax(logits_new)
        objective_new = model.objective(y, logits_new)
        query_perturb += 1

        if objective_new <= 0:
            # print("Attack successful!")
            succ = True
            dist = np.linalg.norm(x_new - x)
            error_counts[predicted_label] += 1
            # save_image(x_new[0], truth, j, f"./results2/attack_success/{args.model}")
            # print("Distance:", dist)
            break
        else:
            if objective_new < objective:
                to_mutate = objective_new < objective
                objective = to_mutate * objective_new + ~to_mutate * objective

                to_mutate = np.reshape(to_mutate, [-1, *[1] * len(x.shape[:-1])])
                x_mutated = to_mutate * x_new + ~to_mutate * x_mutated

    return succ, query_perturb, dist, np.count_nonzero(error_counts), np.sum(error_counts)


def targeted_perturbation(model, x, y, selected_regions, truth, eps, total_queries):

    c, h, w = x.shape[1:]
    # x_mutated = x.copy()
    query = 0
    dist = []
    tag = False
    mean_distance = 0

    num_classes = 10  # 获取模型类别数
    target_classes = [cls for cls in range(num_classes) if cls != truth]
    # target_classes = [cls for cls in selected_labels if cls != truth]
    query_budget = total_queries // len(target_classes)  # 每个类别的查询预算
    success_count = 0  # 记录成功攻击的次数
    mutation_index = np.zeros_like(x)  # 变异向量索引
    best_test_cases = {cls: (None, -np.inf) for cls in range(num_classes)}
    error_counts = np.zeros(num_classes, dtype=int)

    index = 0
    for target_class in target_classes:
        index += 1
        print(f"Targeting class {target_class}...")
        query_perturb = 0  # 当前目标类别的查询次数
        x_mutated = best_test_cases[target_class][0] if best_test_cases[target_class][0] is not None else x.copy()
        # x_mutated = x.copy()
        objective_prev = best_test_cases[target_class][1]
        # objective_prev = -np.inf
        # while query_perturb < query_budget:
        for query_perturb in range(query_budget):
            # **整体扰动**
            if best_test_cases[target_class][0] is None:
                mutation = np.zeros_like(x)
                for start_i, start_j, region in selected_regions:
                    mutation_block = mutation_vector_l2(start_i, start_j, region)
                    mutation[:, :, start_i:min(start_i + region.shape[2], h),
                    start_j:min(start_j + region.shape[3], w)] = mutation_block * np.random.choice([-1, 1],
                                                                                                   size=[x.shape[0], c, 1,
                                                                                                         1])

                x_mutated = np.clip(x_mutated + mutation / np.sqrt(np.sum(mutation ** 2, keepdims=True)) * eps, 0, 1)
                # x_mutated = np.clip(x_mutated + np.clip(mutation / np.sqrt(np.sum(mutation ** 2, keepdims=True)), -eps, eps), 0, 1)
                # x_mutated = np.clip(x_mutated + np.clip(mutation, -eps, eps), 0, 1)
                # **模型预测**
                logits_mutated = model.predict(x_mutated)
                predicted_label = np.argmax(logits_mutated)
                # objectives = {cls: logits_mutated[0][cls] for cls in selected_labels}
                objectives = {cls: logits_mutated[0][cls] for cls in range(num_classes)}
                # **计算目标类别的目标函数值**
                objective_prev = objectives[target_class]
                # objective_prev = model.objective(y, logits_mutated)
                query_perturb += 1

                if predicted_label != truth:
                    # print(f"Attack successful on target {predicted_label}!")
                    error_counts[predicted_label] += 1
                    # error_counts[index] += 1
                    # distance = np.sqrt(np.sum((x_mutated - x) ** 2, axis=(1, 2, 3), keepdims=True))
                    # distance = np.max(np.abs(x_mutated - x), axis=(1, 2, 3), keepdims=True)
                    distance = np.linalg.norm(x_mutated - x) / np.linalg.norm(x)
                    dist.append(distance)

                if predicted_label == target_class:
                    # print(f"Target attack successful on target {target_class}!")
                    success_count += 1
                    tag = True
                    best_test_cases[target_class] = (None, -np.inf)
                    continue

            mutation_index = x_mutated - x

            # mutation insert
            insertion_region = random.choice(selected_regions)
            inser_i, inser_j, inser_region = insertion_region
            new_mutation_block = mutation_vector_l2(inser_i, inser_j, inser_region)
            new_mutations = np.ones([x.shape[0], c, inser_region.shape[2], inser_region.shape[3]]) * np.random.choice(
                [-1, 1], size=[x.shape[0], c, 1, 1])
            mutation_insertion_block = np.zeros(x.shape)
            mutation_insertion_block[:, :, inser_i:inser_i + inser_region.shape[2],
            inser_j:inser_j + inser_region.shape[3]] = 1

            # mutation delete
            mutation_deletion_blocks = np.zeros(x.shape)
            i = 0
            while i < 10:
                delete_region = random.choice(selected_regions)
                start_i, start_j, region = delete_region
                mutation_deletion_blocks[:, :, start_i:start_i + region.shape[2], start_j:start_j + region.shape[3]] = 1
                mutation_index[:, :, start_i:start_i + region.shape[2], start_j:start_j + region.shape[3]] = 0
                i += 1

            distance_insertion_block = np.sqrt(
                np.sum(((x_mutated - x) * mutation_insertion_block) ** 2, axis=(2, 3), keepdims=True))
            distance_mutated_image = np.sqrt(np.sum((x_mutated - x) ** 2, axis=(1, 2, 3), keepdims=True))
            all_blocks = np.maximum(mutation_insertion_block, mutation_deletion_blocks)
            distance_all_blocks = np.sqrt(np.sum((mutation_index * all_blocks) ** 2, axis=(2, 3), keepdims=True))

            old_mutations = mutation_index[:, :, inser_i:inser_i + inser_region.shape[2],
                            inser_j:inser_j + inser_region.shape[3]] / (1e-10 + distance_insertion_block)
            new_mutations = old_mutations + new_mutations
            new_mutations = new_mutations / np.sqrt(np.sum(new_mutations ** 2, axis=(2, 3), keepdims=True)) * (
                    np.maximum(eps ** 2 - distance_mutated_image ** 2, 0) / c + distance_all_blocks ** 2) ** 0.5

            mutation_index[:, :, inser_i:inser_i + inser_region.shape[2],
            inser_j:inser_j + inser_region.shape[3]] = new_mutations + 0

            x_new = x + mutation_index / np.sqrt(np.sum(mutation_index ** 2, axis=(1, 2, 3), keepdims=True)) * eps

            x_new = np.clip(x_new, 0, 1)

            logits_new = model.predict(x_new)
            objectives = {cls: logits_new[0][cls] for cls in range(num_classes)}
            # objectives = {cls: logits_new[0][cls] for cls in selected_labels}
            objective_new = objectives[target_class]
            # objective_new = model.objective(y, logits_new)
            query_perturb += 1

            for cls, obj in objectives.items():
                if obj > best_test_cases[cls][1]:
                    best_test_cases[cls] = (x_new.copy(), obj)

            # **检查是否攻击成功**
            if np.argmax(logits_new) != truth:
                # print(f"Attack successful on target {np.argmax(logits_new)}!")
                error_counts[np.argmax(logits_new)] += 1  # 增加错误计数
                # distance = np.sqrt(np.sum((x_new - x) ** 2, axis=(1, 2, 3), keepdims=True))
                # distance = np.max(np.abs(x_new - x), axis=(1, 2, 3), keepdims=True)
                distance = np.linalg.norm(x_new - x) / np.linalg.norm(x)
                dist.append(distance)

            if np.argmax(logits_new) == target_class:
                # print(f"Target attack successful on target {target_class}!")
                success_count += 1
                tag = True
                best_test_cases[target_class] = (None, -np.inf)
                continue

            if objective_new > objective_prev:
                to_mutate = objective_new > objective_prev
                objective_prev = to_mutate * objective_new + ~to_mutate * objective_prev

                to_mutate = np.reshape(to_mutate, [-1, *[1] * len(x.shape[:-1])])
                x_mutated = to_mutate * x_new + ~to_mutate * x_mutated

        query += query_perturb
        print(f"Target {target_class} attack finished")
        if len(dist) == 0 or np.isnan(dist).all():
            mean_distance = 0
        else:
            mean_distance = np.nanmean(np.asarray(dist))
        # mean_distance = np.mean(np.asarray(dist))
    return query, mean_distance, success_count, np.count_nonzero(error_counts), np.sum(error_counts), tag


def save_image(image, truth, index, save_dir):
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f"{index}_{truth}.png"
    file_path = os.path.join(save_dir, file_name)

    # image = image[0]  # 变为 (C, H, W)

    if image.shape[0] == 1:
        image = image[0]

    # image = np.clip(image * 255, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
    elif image.ndim == 3:
        image_pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))

    image_pil.save(file_path)
    # print(f"Saved attack successful image at: {file_path}")

def deep_region(model, x, y, y_test, logits_clean, objective_clean, eps, n_iters, region_size, region_stop):
    # print(type(x), x.shape)
    # np.random.seed(1)
    c, h, w = x.shape[1:]
    regions = []

    query = 0
    # success = 0
    dist = 0
    succ = False
    # time = 0
    truth = np.argmax(y)
    Q = args.n_iter

    for i in range(0, h, region_size):
        for j in range(0, w, region_size):
            region = x[:, :, i:min(i + region_size, h), j:min(j + region_size, w)]
            regions.append((i, j, region))  # 记录区域及其位置
    # print('finished initializing regions')
    # print(regions)
    # 选择区域
    # select_start = datetime.now()
    region_initial = region_size
    while region_size >= region_stop:
        top_regions, query_selection, mask = select_region(model, x, logits_clean, truth, regions, region_size)
        Q = Q - query_selection
        # print ('finished selecting regions')
        query = query + query_selection
        # print('finished selecting regions')

        if region_size != region_stop:
            regions, new_region_size = split_region(x, top_regions, region_size)
            region_size = new_region_size
            # print('finished splitting regions, region size:{}'.format(region_size))
        else:
            # visualize_selected_regions(x[0], top_regions)
            succ, query_perturb, dist, diversity, error_num = perturb_region(model, x, y, objective_clean, top_regions, truth, eps, n_iters)
            query = query + query_perturb
            if succ:
                return True, query, dist
            else:
                print(f"Failed to attack after {n_iters} iterations.")
                return False, query, dist
            # query, dist, success_count, label_diversity, error_num, tag= targeted_perturbation(model, x, y, top_regions,truth, eps, Q)
            break

    # 随机选择区域
    # top_regions, mask = random_select_region(x, region_stop)
    # succ, query_perturb, dist = deep_rover_single(model, x, y, mask, eps, n_iters, 0.11, 400, 2.0, init_s=12,
    #                                               seed_vect_num=3, radius=5, min_s=7)
    # succ, query_perturb, dist = perturb_region(model, x, y, objective_clean, top_regions, truth, eps, n_iters)
    # query, dist, success_count, label_diversity, error_num, tag = targeted_perturbation(model, x, y, top_regions, truth, eps, Q)

    # return query, dist, success_count, label_diversity, error_num, tag

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters.')
    parser.add_argument('--model', type=str, default='imagenet_inception', choices=models.models_defined,
                        help='name of models')
    parser.add_argument('--exp_results', type=str, default='results2', help='the folder to save experimental results.')
    parser.add_argument('--gpu', type=str, default='0', help='specify the ids of gpus')
    parser.add_argument('--n_ex', type=int, default=1000, help='the number of correctly classified input images to attack.')
    parser.add_argument('--eps', type=float, default=5, help='the L2 distance.')
    parser.add_argument('--n_iter', type=int, default=10000, help='total number of queries.')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = 'cifar10' if 'cifar10' in args.model else 'mnist' if 'mnist' in args.model else 'svhn' if 'svhn' in args.model else 'imagenet'
    timestamp = str(datetime.now())[:10] + '_' + str(datetime.now())[11:13] + '-' + str(datetime.now())[
                                                                                    14:16] + '-' + str(datetime.now())[
                                                                                                   17:19]

    basic_info = '{} model={} dataset={} n_ex={} eps={} n_iter={}'.format(
        timestamp, args.model, dataset, args.n_ex, args.eps, args.n_iter)
    batch_size = data.batch_size_dictionary[dataset]
    region_size = data.region_size_dictionary[dataset]
    region_stop = data.region_stop_dictionary[dataset]

    # n_cls: the number of classes for the dataset
    n_cls = 1000 if dataset == 'imagenet' else 10
    # gpu_memory: the percentage of GPU memory to use
    gpu_memory = 0.99

    log_path = '{}/{}.log'.format(args.exp_results, args.model)
    log = utils.Logger(log_path)
    log.print('Basic Parameters: {}'.format(basic_info))

    n_ex_to_load = 2 * args.n_ex


    if args.model in ['cifar10-defended-trades', 'cifar10-undefended-trades', 'cifar10-vgg16']:
        # x_test, y_test = data.load_cifar10_trades(n_ex_to_load)
        x_test, y_test = data.load_cifar10(n_ex_to_load)
    else:
        if dataset == 'mnist':
            x_test, y_test = data.load_mnist(n_ex_to_load)
            print('mnist')
        else:
            if dataset == 'svhn':
                x_test, y_test = data.load_svhn(n_ex_to_load)
                print('svhn')
            else:
                if dataset == 'imagenet':
                    if args.model != 'imagenet_inception':
                        x_test, y_test = data.load_imagenet(n_ex_to_load)
                    else:
                        # x_test, y_test = data.load_imagenet(n_ex_to_load, size=299)
                        x_test, y_test = data.load_imagenet(n_ex_to_load, size=299)

    print('model name:', args.model)

    if args.model in ['cifar10-defended-trades', 'cifar10-undefended-trades',
                      'svhn-defended-trades', 'svhn-undefended-trades']:
        model = models.CustomModel_Trades(args.model, batch_size, gpu_memory)
    else:
        model = models.CustomModel(args.model, batch_size, gpu_memory)

    logits_clean = model.predict(x_test)

    num_total_input = x_test.shape[0]
    print('sampled', num_total_input, 'input images')

    y_target_onehot = utils.dense_to_onehot(y_test, n_cls=n_cls)
    objective_clean = model.objective(y_target_onehot, logits_clean)

    correctly_classified = logits_clean.argmax(1) == y_test

    log.print('Clean accuracy: {:.2%}'.format(np.mean(correctly_classified)))
    y_target_onehot = utils.dense_to_onehot(y_test, n_cls=n_cls)

    num_total_input = x_test.shape[0]
    print('sampled', num_total_input, 'input images')
    x, y = x_test[correctly_classified], y_target_onehot[correctly_classified]
    x, y = x[:args.n_ex], y[:args.n_ex]
    # x, y = x_test[:args.n_ex], y_target_onehot[:args.n_ex]
    num_total_images = x.shape[0]
    print('select', num_total_images, 'correctly classified images')


    j = 0
    query_all = []
    # success = 0
    success = []
    all_distance = []
    time = []
    label = 0
    error = 0
    TAG = 0

    # save_path = "./data/ImageNet"
    while j < x.shape[0]:
        # image_array = (x[j].transpose(1, 2, 0) * 255).astype(np.uint8)
        # img = Image.fromarray(image_array)
        # img.save(os.path.join(save_path, f'image_{j}.jpeg'))
        print('image_index:', j)
        # query, dist, success_count, label_diversity,error_num, tag = deep_region(model, x[j:j+1],y[j:j+1], y_test, logits_clean[j:j+1], objective_clean[j:j+1], args.eps, args.n_iter, region_size, region_stop)
        succ, query, dist, diversity, error_num = deep_region(model, x[j:j + 1], y[j:j + 1], y_test, logits_clean[j:j + 1], objective_clean[j:j + 1], args.eps, args.n_iter, region_size, region_stop)
        if succ:
            success.append(1)
            query_all.append(query)
            all_distance.append(dist)
            label += diversity
            error += error_num
        # query_all.append(query)
        # success += success_count
        # all_distance.append(dist)
        # # print(distance)
        # label += label_diversity
        # error += error_num
        # if tag:
        #    TAG += 1
        j += 1

    log.print('success rate: {:.2%}'.format(len(success) / num_total_images))
    log.print('query: {}'.format(np.mean(np.asarray(query_all))))
    log.print('Label diversity: {:.2%}'.format(label / len(success)))
    log.print('Average l2 distance: {}'.format(np.mean(np.asarray(all_distance))))
    log.print('Error number: {}'.format(error))
    # log.print('Target success : {}'.format(success))
    # log.print('Success rate: {:.2%}'.format(TAG / args.n_ex))







