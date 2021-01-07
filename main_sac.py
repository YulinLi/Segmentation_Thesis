from env.Snake_Env import SnakeEnv
import algo.sac as sac
import models.models as models
import numpy as np
import os
import sys
import rl_eval
import warnings
warnings.filterwarnings("ignore", category=Warning)
from dataloader.dataloader import TrainDataset, TestDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

batch_size = 64
eval_iter = 10
rl_core = sac.SAC(
    model=[models.PolicyNetGaussian, models.QNet],
    n_actions=2,
    learning_rate=[0.0001, 0.0001],
    reward_decay=0.99,
    memory_size=10000,
    batch_size=batch_size,
    alpha=0.1,
    auto_entropy_tuning=True)

is_train = True
render = False
load_model = False
train_label_pathIn = "/home/mislab/LiYulin/Research/dataset/Flower/train/annotation/"
train_img_pathIn = "/home/mislab/LiYulin/Research/dataset/Flower/train/image/"
test_label_pathIn = "/home/mislab/LiYulin/Research/dataset/Flower/test/annotation/"
test_img_pathIn = "/home/mislab/LiYulin/Research/dataset/Flower/test/image/"
img_path = "image/03.png"
test_gif_path = "test_out/"
train_gif_path = "train_out/"
model_path = "save/"
iou_list = []
train_iou_list = []
iter_list = []
reward_list = []
if not os.path.exists(model_path):
    os.makedirs(model_path)

if load_model:
    print("Load model ...", model_path)
    rl_core.save_load_model("load", model_path)

if __name__ == "__main__":
    train_dataset = TrainDataset(
        train_img_pathIn, train_label_pathIn, transform=False)
    test_dataset = TestDataset(
        test_img_pathIn, test_label_pathIn, transform=False)
    env = SnakeEnv(img=None)
    total_step = 0
    Max_mIOU = 0
    Max_reward = 0
    dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=8)
    # write reward and mIOU
    # end open file
    for eps in range(1000):
        acc_reward = 0.
        for i_batch, sample in enumerate(dataloader):
            state = env.initialize(
                sample['img'].numpy().squeeze(), targets=sample['pts'])
            step = 0
            loss_a = loss_c = 0.

            while(True):
                # Choose action and run
                if is_train:
                    action = rl_core.choose_action(state, eval=False)
                else:
                    action = rl_core.choose_action(state, eval=True)

                state_next, reward = env.step(action.tolist())
                #end = (np.array(done).astype(int)).tolist()

                for i in range(len(state_next)):
                    rl_core.store_transition(
                        state[i], action[i], reward[i], state_next[i])

                # Render environment
                im = env.render(gui=render)
                # Learn the model
                loss_a = loss_c = 0.
                if total_step > batch_size and is_train:
                    loss_a, loss_c = rl_core.learn()
                # End:for end
                step += 1
                total_step += 1

                # Print information
                acc_reward += sum(reward)
                # print(acc_reward)
                print('\rEps:{:3d}/{:4d} /{:4d} /{:6d}| action_0:{:+.2f}| R:{:+.2f}| Loss:[A>{:+.2f} C>{:+.2f}]| Alpha: {:.3f}| R_total:{:.2f}  '
                      .format(eps, i_batch, step, total_step,
                              action[0][0], sum(
                                  reward), loss_a, loss_c, rl_core.alpha,
                              acc_reward), end='')

                state = state_next.copy()
                # if done or step > 3:
                if step > 2:
                    break
            # End learning
            # Start eval
            if((i_batch+1) % eval_iter == 0 or i_batch == len(dataloader)):
                reward_list.append(acc_reward)
                acc_reward = 0.
                iter_list.append(total_step/step)
                # Save the best model
                if acc_reward > Max_reward:
                    if is_train:
                        print("Save model to " + model_path)
                        rl_core.save_load_model("save", model_path)
                    # output GIF
                total_iou_score = 0
                total_train_iou_score =0
                test_dataloader = DataLoader(
                    test_dataset, batch_size=1, shuffle=None, num_workers=4)
                print('\n')
                for j_batch, sample in enumerate(test_dataloader):
                    img_test, points_test = sample['img'].numpy(
                    ).squeeze(), sample['pts']
                    iouscore = rl_eval.run(rl_core,  total_eps=1, is_train=True, img=img_test,
                                           points=points_test, gif_path=test_gif_path, gif_name="sac_" +str(eps).zfill(3) + "_" +str(i_batch).zfill(5)+"_"+str(j_batch))
                    print('\rimage_idx: {:2d}| IOU: {:2f}'.format(
                        j_batch, iouscore), end='')
                    total_iou_score += iouscore

                mIOU = total_iou_score/len(test_dataset)
                print('\niter:{:2d}|mIOU: {:2f}'.format(i_batch, mIOU))

                for j_batch, sample in enumerate(dataloader):
                    img_test, points_test = sample['img'].numpy(
                    ).squeeze(), sample['pts']
                    train_iouscore = rl_eval.run(rl_core,  total_eps=1, is_train=True, img=img_test,
                                           points=points_test, gif_path=train_gif_path, gif_name="sac_" +str(eps).zfill(3) + "_" +str(i_batch).zfill(5)+"_"+str(j_batch))
                    print('\rimage_idx: {:2d}| IOU: {:2f}'.format(
                        j_batch, train_iouscore), end='')
                    total_train_iou_score += train_iouscore
                train_mIOU = total_train_iou_score/len(train_dataset)
                print('\niter:{:2d}|mIOU: {:2f}'.format(i_batch, train_mIOU))

                fig, ax = plt.subplots(nrows=3, ncols=1)
                # draw reward change
                ax[0].plot(iter_list, reward_list)
                ax[0].set_title("Training Total Reward")
                ax[0].set(xlabel='iter', ylabel='reward')
                # draw train IOU
                train_iou_list.append(train_mIOU)
                ax[1].plot(np.array(iter_list),  np.array(train_iou_list))
                ax[1].set_title("Training IOU")
                ax[1].set(xlabel='iter', ylabel='mIOU')
                # draw test IOU
                iou_list.append(mIOU)
                ax[2].plot(np.array(iter_list),  np.array(iou_list))
                ax[2].set_title("Testing IOU")
                ax[2].set(xlabel='iter', ylabel='mIOU')
                fig.tight_layout()
                plt.savefig("iou_image/test_m_iou_" +
                            str(i_batch*eps).zfill(5)+".jpg")
                plt.close(fig)
