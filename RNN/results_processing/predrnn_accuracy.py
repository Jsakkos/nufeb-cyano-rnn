import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from glob import glob

def predrnn_accuracy(DATA_DIR, SAVE_DIR, LOC, show_plots, save_plots, layer_str):
    LOCS = [LOC] # This is legacy to be able to compare various layers and lrs

    all_mse = []
    all_ssim = []
    all_psnr = []
    all_lpips = []
    all_tr = []
    all_tr_itr = []
    all_te_itr = []

    print("Prediction accuracy (mse, etc.): ")
    print("------------------------------")
    for LOC in LOCS:
        BASE = DATA_DIR + LOC + ".log"
        print("Read: ", BASE)
        mse = []
        ssim = []
        psnr = []
        lpips = []

        training_loss = []
        training_itr = []
        test_itr = []
        params = ""

        with open(BASE, "r") as file:
            for line in file.readlines()[12:]:
                if "Namespace" in line:
                    params += line
                if "training loss" in line:
                    training_loss.append(float(line.split(" ")[-1].strip()))
                if "itr" in line:
                    training_itr.append(int(line.split(" ")[-1].strip()))
                if "test" in line:
                    test_itr.append(training_itr[-1])
                if "mse per seq" in line:
                    mse.append(float(line.split(" ")[-1].strip()))
                if "ssim per frame" in line:
                    ssim.append(float(line.split(" ")[-1].strip()))
                if "psnr per frame" in line:
                    psnr.append(float(line.split(" ")[-1].strip()))
                if "lpips per frame" in line:
                    lpips.append(float(line.split(" ")[-1].strip()))

        mse = np.array(mse).T
        ssim = np.array(ssim).T
        psnr = np.array(psnr).T
        lpips = np.array(lpips).T
        tr_loss = np.array(training_loss).T
        tr_itr = np.array(training_itr).T
        test_itr = np.array(test_itr).T

        all_mse.append(mse)
        all_ssim.append(ssim)
        all_psnr.append(psnr)
        all_lpips.append(lpips)
        if "test" in LOC:
            all_tr.append(np.zeros(1))
            all_tr_itr.append(np.zeros(1))
            all_te_itr.append(np.zeros(1))
        else:
            all_tr.append(tr_loss)
            all_tr_itr.append(tr_itr)
            all_te_itr.append(test_itr)

    # Save parameters
    params = [p.strip() for p in params.split(",")]
    max_iterations = 0
    test_interval = 0
    for p in params:
        if "max_iterations" in p:
            max_iterations = int(p.split("=")[-1])
        if "test_interval" in p:
            test_interval = int(p.split("=")[-1])
    # Plots
    def make_plots(all_items, all_xticks, ylabel):
        plt.figure(figsize=(8, 6))
        colors = cm.get_cmap("Set1").colors
        if len(all_items) == 1:
            # plt.scatter(all_xticks[0], all_items[0])
            plt.plot(all_xticks[0], all_items[0], c=colors[0], lw=3, label=layer_str)
        else:
            for i, sub_item in enumerate(all_items):
                plt.plot(all_xticks[i], sub_item, label=layer_str)
        plt.xlabel("Training iteration", fontsize=24)
        plt.ylabel(ylabel, fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.legend()
        # plt.title(LOC[:-len(layer_str)])
        if save_plots:
            if os.path.exists(SAVE_DIR + LOC) == False:
                os.mkdir(SAVE_DIR + LOC)
            save_loc = SAVE_DIR + LOC + "/{}.png".format(ylabel)
            print("Save: {}".format(save_loc))
            plt.savefig(save_loc)
        if show_plots:
            plt.show()
        else:
            plt.close()

    make_plots(all_mse, all_te_itr, "MSE")
    make_plots(all_ssim, all_te_itr, "ssim")
    make_plots(all_psnr, all_te_itr, "psnr")
    make_plots(all_lpips, all_te_itr, "LPIPS")
    make_plots(all_tr, all_tr_itr, "training_loss")
