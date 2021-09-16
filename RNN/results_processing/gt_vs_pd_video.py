import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def gt_vs_pd_videos(DATA_DIR, SAVE_DIR, LOC, num_frames, show_videos, save_videos, jump_size=2):
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

    print("Comparison videos:")
    print("-------------------------")
    dir_num = batches[-1]
    print("Read: ", DATA_DIR + LOC + "/{}".format(dir_num))
    for test_num in tests:
        batch_dir = DATA_DIR + LOC + "/{}/{}".format(dir_num, test_num)
        # Load the images
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
        # for file_num in range(1, num_frames):
        #     if len(plt.imread(batch_dir + "/gt" + str(file_num) + ".png").shape) == 2:
        #         gt_ims.append(plt.imread(batch_dir + "/gt" + str(file_num) + ".png"))
        #         pd_ims.append(plt.imread(batch_dir + "/pd" + str(file_num + num_frames//2) + ".png"))
        #     else:
        #         gt_ims.append(plt.imread(batch_dir + "/gt" + str(file_num) + ".png")[:, :, ::-1])
        #         pd_ims.append(plt.imread(batch_dir + "/pd" + str(file_num + num_frames//2) + ".png")[:, :, ::-1])

        # Brighten images
        gt_ims = np.array(img_array)
        pd_ims = np.array(p_img_array)
        gt_ims *= 1/np.max(gt_ims)
        pd_ims *= 1/np.max(pd_ims)
        ims = [gt_ims, pd_ims]

        # Make animation
        fig = plt.figure(figsize=(12, 6))
        axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
        axes[0].set_title("Ground truth")
        axes[1].set_title("Predicted")
        plots = [axes[0].imshow(gt_ims[0]), axes[1].imshow(pd_ims[0])]
        plt.xticks([])
        plt.yticks([])

        def anim_func(i, plots):
            for j in range(len(plots)):
                plots[j].remove()
            for j in range(len(plots)):
                plots[j] = axes[j].imshow(ims[j][i])
            return plots

        results_anim = anim.FuncAnimation(fig, anim_func, range(1, len(gt_ims)), fargs=[plots], blit=True)
        if save_videos:
            save_loc = SAVE_DIR + LOC + "_iteration-{}_batch-{}.mp4".format(dir_num, test_num)
            print("Save: {}".format(save_loc))
            results_anim.save(save_loc)
        if show_videos:
            plt.show()
        plt.close(fig)
