import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def gt_vs_pd_panels(DATA_DIR, SAVE_DIR, LOC, num_frames, show_plots, save_plots, jump_size=2):
    batches = []
    print(DATA_DIR + LOC)
    for batch_dir in glob(DATA_DIR + LOC + "/*"):
        print(batch_dir)
        if "test" in batch_dir:
            batches.append(batch_dir.split("/")[-1])
        else:
            batches.append(int(batch_dir.split("/")[-1]))
    batches.sort()
    tests = []
    for test_dir in glob(DATA_DIR + LOC + "/{}/*".format(batches[0])):
        tests.append(int(test_dir.split("/")[-1]))
    tests.sort()

    print("Panel plots:")
    print("-------------------------")
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

        # Brighten images
        img_array = np.array(img_array)
        p_img_array = np.array(p_img_array)
        img_array *= 1/img_array.max()
        p_img_array *= 1/p_img_array.max()

        ims = [img_array, p_img_array]
        # num_ims = len(img_array[num_frames//2:num_frames+1:jump_size])
        num_ims = len(img_array[:num_frames+1:jump_size])
        titles = ["True -- Frame: ", "Predicted -- Frame: "]

        fig = plt.figure(figsize=(4*num_ims, 8))
        for i in range(2):
            for j in range(num_ims):
                plt.subplot(2, num_ims, i*num_ims+j+1)
                if len(ims[0][0].shape) == 2:
                    # plt.imshow(ims[i][num_frames//2 + j*jump_size], cmap="gray")
                    plt.imshow(ims[i][j*jump_size], cmap="gray")
                else:
                    # plt.imshow(ims[i][num_frames//2 + j*jump_size])
                    plt.imshow(ims[i][j*jump_size])
                # plt.title(titles[i] + str(num_frames//2 + j*jump_size), fontsize=18)
                plt.title(titles[i] + str(j*jump_size), fontsize=18)
                plt.xticks([])
                plt.yticks([])

        plt.tight_layout()
        if save_plots:
            save_loc = SAVE_DIR + LOC + "_iteration-{}_batch-{}.png".format(dir_num, test_num)
            print("Save: ", save_loc)
            plt.savefig(save_loc)
        if show_plots:
            plt.show()
        plt.close()

