import cv2
import numpy as np
import util.Snake_tool as tool
import math
import random
import torch


class SnakeContourPoint:
    def __init__(self, idx):
        self.idx = idx

    def init_state(self, r, theta):
        self.r = r
        self.theta = theta


class SnakeEnv:
    def __init__(self, img):
        self.img = img
        self.targets = []
        # self.img = np.asarray(self.img).astype(float)
        self.w = 128  # self.width
        self.init_scp_num = 8  # To set init scp number
        self.center = (64, 64)  # Polar coordinate center
        self.inter_theta = 2*np.pi/self.init_scp_num
        self.init_r = [60]*self.init_scp_num  # Init SCP radius
        #self.init_r = np.random.randint(24, 40, self.init_scp_num)
        #self.init_r = 60
        self.target_dist = []*self.init_scp_num
        self.scp = []
        self.stop = [False]*self.init_scp_num
        self.out_of_bound = [False]*self.init_scp_num
        self.iou_score = 0.0
        self.iter = 0

    def initialize(self, img, targets=None, is_train=True):
        # Init Snake Point, degree interval 0,45, ... 315 degree
        # Snake_point_list = []
        self.img = img
        if(is_train == True):
            self.targets = targets
        else:
            self.targets = torch.zeros(self.init_scp_num, 2)

        # Clear init
        self.scp.clear()
        self.stop = [False]*self.init_scp_num
        self.out_of_bound = [False]*self.init_scp_num
        # self.area_color_var.clear()

        # random init inner or outter
        """if(self.iter == 0):
            #self.init_r = np.random.randint(44, 52, self.init_scp_num)
            self.init_r = [48]*self.init_scp_num
            self.iter = 1
        else:
            #self.init_r = np.random.randint(12, 20, self.init_scp_num)
            self.init_r = [16]*self.init_scp_num
            self.iter = 0"""
        
        # Init SCP points
        for i in range(self.init_scp_num):
            scp_temp = SnakeContourPoint(i)
            scp_temp.init_state(self.init_r[i], self.inter_theta*i)
            self.scp.append(scp_temp)

        for i in range(self.init_scp_num):
            x, y = tool.P2C(self.scp[i].r, self.scp[i].theta, self.center)
            self.target_dist.append(np.sqrt(
                (x - self.targets[i][0])**2 + (y - self.targets[i][1])**2))
        # return state
        state = []
        for i in range(self.init_scp_num):
            temp = self._construct_state(
                self.img, self.scp[(i-1+self.init_scp_num) %
                                   self.init_scp_num],
                self.scp[(i+self.init_scp_num) % self.init_scp_num],
                self.scp[(i+1+self.init_scp_num) % self.init_scp_num])
            state.append(temp)
        """, self.snake2, self.snake3, self.snake4"""
        return state

    def step(self, action):
        # Reward weight
        w_dif = 1
        w_same = -1
        # 0.Last state snake point position
        pre_scp = self.scp
        # 1.Update snake point position
        for i in range(self.init_scp_num):
            if(self.stop[i] != True):
                self.scp[i].r += 10*action[i][0]
                if(self.scp[i].r > self.w/2 -1):
                    self.out_of_bound[i] = True
                    self.scp[i].r = self.w/2 - 1 
                if(self.scp[i].r < -self.w/2+1):
                    self.out_of_bound[i] = True
                    self.scp[i].r = -self.w/2+1
        # 2. Count reward
        # Init List
        curr_target_dist = []
        curr_target_dist.clear()
        reward_dist = []
        reward_dist.clear()

        # 2-1, 2-2: Calculate Var and Area Reward
        for i in range(self.init_scp_num):
            x, y = tool.P2C(self.scp[i].r, self.scp[i].theta, self.center)
            # supervised target
            cur_dist = np.sqrt(
                (x - self.targets[i][0])**2 + (y - self.targets[i][1])**2)
            curr_target_dist.append(cur_dist)
            reward_dist.append(self.target_dist[i] - curr_target_dist[i])

        # 2-4: Action Penalty
        reward_lazy_penalty = []
        reward_lazy_penalty.clear()

        for i in range(self.init_scp_num):
            if(reward_dist[i] < 0 and curr_target_dist[i] > 5):
                reward_lazy_penalty.append(-0.1)
            else:
                reward_lazy_penalty.append(0)
        # 2-5: Total Reward
        """reward_list = [np.array([i*w_var for i in reward_area_color_var]),
                       np.array([i*w_area_size for i in reward_area_size]), reward_lazy_penalty]"""
        reward = np.add(
            np.array([i*0.1 for i in reward_dist]), np.array(reward_lazy_penalty))

        #reward = sum(reward_list)
        """print('\n')
        print(reward_dif)
        print(reward_same)
        print(reward_list)
        print(reward)"""
        # Terminal State
        #done = [False]*self.init_scp_num
        for i in range(self.init_scp_num):
            # out of bound
            if (self.out_of_bound[i] == True and self.stop[i] == False):
                reward[i] = -10
                #done[i] = True
                self.stop[i] = True
            if (np.abs(curr_target_dist[i]) < 2 and self.stop[i] == False):
                reward[i] = 10
                #done[i] = True
                self.stop[i] = True
            if(self.stop[i] == True):
                reward[i] = 0
            

        self.target_dist = curr_target_dist
        # update cur color var
        # 3. Construct next state
        # return state
        state_next = []
        state_next.clear()
        for i in range(self.init_scp_num):
            temp = self._construct_state(
                self.img, pre_scp[(i-1+self.init_scp_num) %
                                  self.init_scp_num],
                self.scp[(i+self.init_scp_num) % self.init_scp_num],
                pre_scp[(i+1+self.init_scp_num) % self.init_scp_num])
            state_next.append(temp)

        """, self.snake2, self.snake3, self.snake4"""
        return state_next, reward  # , done

    def render(self, gui=True):
        img_ = self.img.copy()
        # x y coordinate
        scp_x = []
        scp_y = []
        pts = []

        for i in range(self.init_scp_num):
            x, y = tool.P2C(self.scp[i].r, self.scp[i].theta, self.center)
            """
            print('scp_{:d},r:{:3f},theta:{:3f}'.format(
                i, self.scp[i].r, self.scp[i].theta))
            """
            if(x > self.w):
                draw_x = self.w
            elif(x < 0):
                draw_x = 0
            else:
                draw_x = x

            if(y > self.w):
                draw_y = self.w
            elif(y < 0):
                draw_y = 0
            else:
                draw_y = y

            pts.append([draw_x, draw_y])
            # draw SCP
            cv2.circle(img_, (int(draw_x), int(
                draw_y)), 2, (0, 0, 255), 3)

        thickness = 2
        isClosed = True
        color = (0, 0, 255)

        # draw train poly
        pts = np.array(pts, np.int32)
        img_ = cv2.polylines(img_, [pts], isClosed, color, thickness)
        # draw mask poly
        mask_train = np.zeros(img_.shape, dtype=np.uint8)
        mask_train = cv2.fillPoly(mask_train, [pts], (255, 255, 255))
        mask_train = cv2.cvtColor(mask_train, cv2.COLOR_BGR2GRAY)
        # draw gt poly
        mask_gt = np.zeros(img_.shape, dtype=np.uint8)
        target_pts = np.array(self.targets, np.int32)
        mask_gt = cv2.fillPoly(mask_gt, [target_pts], (255, 255, 255))
        mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
        # or and
        inter = np.logical_and(mask_train, mask_gt)
        union = np.logical_or(mask_train, mask_gt)
        iou_score = np.sum(inter)/np.sum(union)
        self.iou_score = iou_score
        if gui:
            cv2.imshow("Snake Segmentation", img_)
            k = cv2.waitKey(1)
        return img_

    def get_iou_score(self):
        return self.iou_score

    def _construct_state(self, img, SCP_l, SCP, SCP_r):
        state = []
        # rotate to same direction
        rotated = tool.rotate_image(img, math.degrees(SCP.theta))
        l_x, l_y = tool.P2C(
            SCP_l.r, 2*np.pi*(self.init_scp_num-1)/self.init_scp_num, self.center)
        x, y = tool.P2C(SCP.r, 0, self.center)
        #r_x, r_y = tool.P2C(SCP_r.r, 2*np.pi/self.init_scp_num, self.center)
        l_x, l_y = tool.P2C(
            SCP_l.r, SCP_l.theta, self.center)
        x, y = tool.P2C(SCP.r, SCP.theta, self.center)
        r_x, r_y = tool.P2C(SCP_r.r, SCP_r.theta, self.center)
        #
        state.append(rotated/255)
        state.append(l_x)
        state.append(l_y)
        state.append(x)
        state.append(y)
        state.append(r_x)
        state.append(r_y)

        return state


if __name__ == "__main__":
    env = SnakeEnv()
    env.initialize()
    print(self.inter_theta)
    env = SnakeEnv()
    for i in range(10):
        env.initialize()
        while(True):
            action = 2*np.random.random(2)-1
            sn, r, end = env.step(action)
            print(sn[20:23], r, end)
            print(len(sn))
            env.render()
            if end:
                break
