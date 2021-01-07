import algo.sac as sac
from env.Snake_Env import SnakeEnv
import models.models as models
import numpy as np
import os
from PIL import Image
import cv2
from dataloader.dataloader import TrainDataset, TestDataset
from torch.utils.data import Dataset, DataLoader


def run(rl_core, total_eps=2, message=True, render=False, is_train=True,resolution=1,
        img=None, points=None, gif_path="out/", gif_name="img_99.img"):
    gif_path = gif_path + gif_name[4:7] + '/'
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    images = []
    env = SnakeEnv(img)
    iou_score = 0
    for eps in range(total_eps):
        step = 0
        max_success_rate = 0
        success_count = 0

        state = env.initialize(img, points, is_train=True)
        r_eps = []
        acc_reward = 0.

        while(True):
            # Choose action and run
            action = rl_core.choose_action(state, eval=True)
            state_next, reward = env.step(action.tolist())
            im = env.render(gui=render)
            im_pil = Image.fromarray(cv2.cvtColor(
                np.uint8(im), cv2.COLOR_BGR2RGB))
            images.append(im_pil)

            # Record and print information
            r_eps.append(reward)
            acc_reward += reward[0]

            state = state_next.copy()
            step += 1
            if step > 5:
                # images.append(im_pil)
                final_mask = env.render(gui=render)
                iou_score += env.get_iou_score()                
                break
    iou_score /= total_eps
    #print("Save evaluation img ...")
    if gif_path is not None:
        images[0].save(gif_path+gif_name +  ".gif",
                       save_all=True, append_images=images[0:], optimize=True, duration=40, loop=0)
        # if(is_train == False):
        cv2.imwrite(gif_path+gif_name + '.jpg', final_mask)
    return iou_score


if __name__ == "__main__":
    import algo.sac
    rl_core = sac.SAC(
        model=[models.PolicyNetGaussian, models.QNet],
        n_actions=2,
        learning_rate=[0.0001, 0.0001],
        reward_decay=0.99,
        memory_size=10000,
        batch_size=8,
        alpha=0.1,
        auto_entropy_tuning=True)

    test_label_pathIn = "C:/Users/Mislab/Desktop/Research/dataset/LFW/test/annotation/"
    test_gif_pathIn = "C:/Users/Mislab/Desktop/Research/dataset/LFW/test/image/"
    test_dataset = TestDataset(test_gif_pathIn, test_label_pathIn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=None, num_workers=0)

    rl_core.save_load_model("load", "save/")
    for i_batch, sample in enumerate(test_dataloader):
        iouscore = run(rl_core, total_eps=1, is_train=False, resolution=2,img=sample['img'].numpy().squeeze(),
                       points=sample['pts'], gif_name="sac_test"+str(i_batch).zfill(4)+"_"+str(i_batch) + ".gif")
