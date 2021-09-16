import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from glob import glob
from itertools import product

def nufeb_populations(DATA_DIR, SAVE_DIR, LOC, num_frames, show_plots, save_plots):
    batches = []
    for batch_dir in glob(DATA_DIR + LOC + "/*"):
        if "test" in batch_dir:
            batches.append(batch_dir.split("/")[-1])
        else:
            batches.append(int(batch_dir.split("/")[-1]))
    batches.sort()
    tests = []
    for test_dir in glob(DATA_DIR + LOC + "/{}/*".format(batches[0])):
        tests.append(int(test_dir.split("/")[-1]))
    tests.sort()

    print("Population curves:")
    print("-------------------------")
    plt.figure(figsize=(12, 8))
    dir_num = batches[-1]
    print("Read: ", DATA_DIR + LOC + "/{}".format(dir_num))
    for test_num in tests:
        batch_dir = DATA_DIR + LOC + "/{}/{}".format(dir_num, test_num)
        img_array = []
        p_img_array = []
        gt_img = []
        pd_img = []

        for file_num in range(1,num_frames):
            gt_img.append(batch_dir + "/gt" + str(file_num) + ".png")

        for file_num in range(num_frames//2 + 1, 3*num_frames//2):
            pd_img.append(batch_dir + "/pd" + str(file_num) + ".png")

        for gt_file, pd_file in zip(gt_img, pd_img):
            if len(plt.imread(gt_file).shape) == 2:
                img_array.append(plt.imread(gt_file))
                p_img_array.append(plt.imread(pd_file))
            else:
                img_array.append(plt.imread(gt_file)[:, :, ::-1])
                p_img_array.append(plt.imread(pd_file)[:, :, ::-1])


        # Threshold
        if len(np.array(img_array).shape) == 4:
            img_array = np.sum(np.array(img_array), axis=-1)
            p_img_array = np.sum(np.array(p_img_array), axis=-1)

        gt_masses = np.sum(img_array, axis=(1, 2))
        pd_masses = np.sum(p_img_array, axis=(1, 2))

        linestyles = list(lines.lineStyles.keys())[:-3]
        colors = plt.rcParams['axes.prop_cycle'].by_key()["color"]
        styles = list(product(colors, linestyles))
        linelen = len(linestyles)
        # plt.plot(gt_masses, label="Test batch {}: true".format(test_num), linestyle=linestyles[(test_num-1) % linelen])
        # plt.plot(pd_masses, label="Test batch {}: predicted".format(test_num), linestyle=linestyles[(test_num-1) % linelen])
        style = styles[(test_num-1) % len(styles)]
        plt.plot(gt_masses, label="Test batch {}: true".format(test_num), color=style[0], linestyle=style[1], alpha=.5)
        plt.plot(pd_masses, label="Test batch {}: predicted".format(test_num), color=style[0], linestyle=style[1])
    plt.title("Mass comparison - {} - Iteration {}".format(LOC, dir_num))
    plt.ylabel("Mass (pixels)")
    plt.xlabel("Frame number")
    plt.legend(fontsize=8)

    if save_plots:
        plot_save = SAVE_DIR + LOC + "_total-mass.png"
        print("Save: ", plot_save)
        plt.savefig(plot_save)
    if show_plots:
        plt.show()
    plt.close()
