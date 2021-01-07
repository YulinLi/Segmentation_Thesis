import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import util.Snake_tool as tool

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC():
    def __init__(
        self,
        model,
        n_actions,
        learning_rate=[1e-4, 2e-4],
        reward_decay=0.98,
        replace_target_iter=300,
        memory_size=5000,
        batch_size=64,
        tau=0.01,
        alpha=0.5,
        auto_entropy_tuning=True,
        criterion=nn.MSELoss()
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.criterion = criterion
        self._build_net(model[0], model[1])
        self.init_memory()

    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet().to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Evaluation Critic Network (new)
        self.critic = cnet().to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        # Target Critic Network (old)
        self.critic_target = cnet().to(device)
        self.critic_target.eval()

        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.Tensor(self.n_actions).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=0.0001)

    def save_load_model(self, op, path):
        anet_path = path + "sac_anet.pt"
        cnet_path = path + "sac_cnet.pt"
        if op == "save":
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
        elif op == "load":
            self.critic.load_state_dict(
                torch.load(cnet_path, map_location=device))
            self.critic_target.load_state_dict(
                torch.load(cnet_path, map_location=device))
            self.actor.load_state_dict(
                torch.load(anet_path, map_location=device))

    def choose_action(self, s, eval=False):
        img = s
        # current coordinate
        # s_coord = torch.FloatTensor(s[:, 1:].astype(float)).to(device)
        s = np.array(s)
        s_coord = s[:, 1:]
        # 1.full image
        s_img = torch.FloatTensor(np.stack(np.array(s)[:, 0])).to(device)
        s_img = s_img.permute(0, 3, 1, 2)
        # 2.gaussian image list
        scp_num = int(len(s[:]))

        """for i in range(scp_num):
            # create gaus left
            temp_g_img_left = tool.get_gaussian(
                s_coord[i][0], s_coord[i][1], 4, 32, 128)
            temp_g_img = tool.get_gaussian(
                s_coord[i][2], s_coord[i][3], 4, 32, 128)
            temp_g_img_right = tool.get_gaussian(
                s_coord[i][4], s_coord[i][5], 4, 32, 128)
            # convert np 2 tensor
            temp_g_img_left = torch.FloatTensor(
                np.expand_dims(temp_g_img_left, 0)).cuda()
            temp_g_img = torch.FloatTensor(
                np.expand_dims(temp_g_img, 0)).cuda()
            temp_g_img_right = torch.FloatTensor(
                np.expand_dims(temp_g_img_right, 0)).cuda()
            # B, H, W, C-> B, C, H, W
            temp_g_img_left = temp_g_img_left.permute(0, 3, 1, 2)
            temp_g_img = temp_g_img.permute(0, 3, 1, 2)
            temp_g_img_right = temp_g_img_right.permute(0, 3, 1, 2)
            # add into gaus list
            g_img = torch.cat(
                (temp_g_img_left, temp_g_img, temp_g_img_right), dim=1)
            if(i == 0):
                # stacked image
                s_g_img = g_img
            else:
                s_g_img = torch.cat((s_g_img, g_img), dim=0)
        s_g_img = torch.cat((s_img, s_g_img), dim=1)"""
        """
        # Test block start-gaussian image
        img = np.asarray(s_g_img).astype(float)
        cv2.imshow("image", img)
        cv2.waitKey()
        # Test block end
        """
        # 3.cropped image
        # # convert coordinate to grid
        crop_img = []
        for i in range(scp_num):
            img_ = np.array(s)[i, 0]
            temp_crop_img_left = tool.cropped(
                s[i][0], s_coord[i][0], s_coord[i][1], 64, 128)
            temp_crop_img_left = torch.FloatTensor(
                np.expand_dims(temp_crop_img_left, 0)).cuda()
            temp_crop_img = tool.cropped(
                s[i][0], s_coord[i][2], s_coord[i][3], 64, 128)
            temp_crop_img = torch.FloatTensor(
                np.expand_dims(temp_crop_img, 0)).cuda()
            temp_crop_img_right = tool.cropped(
                s[i][0], s_coord[i][4], s_coord[i][5], 64, 128)
            temp_crop_img_right = torch.FloatTensor(
                np.expand_dims(temp_crop_img_right, 0)).cuda()
            # B, H, W, C-> B, C, H, W
            temp_crop_img_left = temp_crop_img_left.permute(0, 3, 1, 2)
            temp_crop_img = temp_crop_img.permute(0, 3, 1, 2)
            temp_crop_img_right = temp_crop_img_right.permute(0, 3, 1, 2)
            # add into gaus list
            crop_img = torch.cat(
                (temp_crop_img_left, temp_crop_img, temp_crop_img_right), dim=1)
            if(i == 0):
                # stacked crop img
                s_crop_img = crop_img
            else:
                s_crop_img = torch.cat((s_crop_img, crop_img), dim=0)
        s_coord = torch.FloatTensor(s_coord.astype(float)).to(device)
        if eval == False:
            action, _, _ = self.actor.sample(
                s_img, s_crop_img, s_coord)
        else:
            _, _, action = self.actor.sample(
                s_img, s_crop_img, s_coord)
        # Output [B,deltaR] action
        action = action.cpu().detach().numpy()

        return action

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s": [], "a": [], "r": [], "sn": [], "end": []}

    def store_transition(self, s, a, r, sn):
        if self.memory_counter <= self.memory_size:
            self.memory["s"].append(s)
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["sn"].append(sn)
            # self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            self.memory["s"][index] = s
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["sn"][index] = sn
            #self.memory["end"][index] = end

        self.memory_counter += 1

    def soft_update(self, TAU=0.01):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_(
                    (1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)

        s_batch = [self.memory["s"][index] for index in sample_index]
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        sn_batch = [self.memory["sn"][index] for index in sample_index]
        #end_batch = [self.memory["end"][index] for index in sample_index]

        # Construct torch tensor
        s_batch_np = np.array(s_batch)
        s_coord = s_batch_np[:, 1:]
        # current image state
        s_img = torch.FloatTensor(np.stack(np.array(s_batch)[:, 0])).to(device)
        """
        # Test block start-gaussian image
        for i in range(16):
            img = s_img[i, :, :, :].cpu().detach().numpy()
            cv2.imshow("image", img)
            cv2.waitKey()
        # Test block end
        """
        s_img = s_img.permute(0, 3, 1, 2)
        B, _, _, _ = s_img.shape
        """# gaussian image
        # 1
        for i in range(B):
            # create gaus left
            temp_g_img_left = tool.get_gaussian(
                s_coord[i][0], s_coord[i][1], 4, 32, 128)
            temp_g_img = tool.get_gaussian(
                s_coord[i][2], s_coord[i][3], 4, 32, 128)
            temp_g_img_right = tool.get_gaussian(
                s_coord[i][4], s_coord[i][5], 4, 32, 128)
            # convert np 2 tensor
            temp_g_img_left = torch.FloatTensor(
                np.expand_dims(temp_g_img_left, 0)).cuda()
            temp_g_img = torch.FloatTensor(
                np.expand_dims(temp_g_img, 0)).cuda()
            temp_g_img_right = torch.FloatTensor(
                np.expand_dims(temp_g_img_right, 0)).cuda()
            # B, H, W, C-> B, C, H, W
            temp_g_img_left = temp_g_img_left.permute(0, 3, 1, 2)
            temp_g_img = temp_g_img.permute(0, 3, 1, 2)
            temp_g_img_right = temp_g_img_right.permute(0, 3, 1, 2)
            # add into gaus list
            g_img = torch.cat(
                (temp_g_img_left, temp_g_img, temp_g_img_right), dim=1)
            if(i == 0):
                # stacked image
                s_g_img = g_img
            else:
                s_g_img = torch.cat((s_g_img, g_img), dim=0)
        s_g_img = torch.cat((s_img, s_g_img), dim=1)"""
        """
        print(s_coord[0, 0].cpu().detach().numpy())
        print(s_coord[0, 1].cpu().detach().numpy())
        g_img = torch.squeeze(s_g_img).cpu().detach().numpy()
        print(g_img.shape)
        cv2.imshow("image", g_img)
        cv2.waitKey()
        """
        # cropped image
        for i in range(B):
            # 1.cropped
            temp_crop_img_left = tool.cropped(
                s_batch_np[i][0], s_coord[i][0], s_coord[i][1], 64, 128)
            temp_crop_img_left = torch.FloatTensor(
                np.expand_dims(temp_crop_img_left, 0)).cuda()
            temp_crop_img = tool.cropped(
                s_batch_np[i][0], s_coord[i][2], s_coord[i][3], 64, 128)
            temp_crop_img = torch.FloatTensor(
                np.expand_dims(temp_crop_img, 0)).cuda()
            temp_crop_img_right = tool.cropped(
                s_batch_np[i][0], s_coord[i][4], s_coord[i][5], 64, 128)
            temp_crop_img_right = torch.FloatTensor(
                np.expand_dims(temp_crop_img_right, 0)).cuda()
            # 3.permute : B, H, W, C-> B, C, H, W
            temp_crop_img_left = temp_crop_img_left.permute(0, 3, 1, 2)
            temp_crop_img = temp_crop_img.permute(0, 3, 1, 2)
            temp_crop_img_right = temp_crop_img_right.permute(0, 3, 1, 2)
            # 4.concat 3 of them in channel dimension
            #crop_img = temp_crop_img
            crop_img = torch.cat(
                (temp_crop_img_left, temp_crop_img, temp_crop_img_right), dim=1)
            # 5. concat batch of them in batch dimension
            if(i == 0):
                # stacked crop img
                s_crop_img = crop_img
            else:
                s_crop_img = torch.cat((s_crop_img, crop_img), dim=0)
        # action
        a_ts = torch.FloatTensor(np.array(a_batch)).to(device)
        # reward
        r_ts = torch.FloatTensor(np.array(r_batch)).to(
            device).view(self.batch_size, 1)
        # next img state
        sn_batch_np = np.array(sn_batch)
        sn_coord = sn_batch_np[:, 1:]
        # next state image
        sn_img = torch.FloatTensor(
            np.stack(np.array(sn_batch)[:, 0])).to(device)
        sn_img = sn_img.permute(0, 3, 1, 2)
        B, _, _, _ = sn_img.shape
        # gaussian image
        # 1
        """for i in range(B):
            # create gaus left
            temp_g_img_left = tool.get_gaussian(
                sn_coord[i][0], sn_coord[i][1], 4, 32, 128)
            temp_g_img = tool.get_gaussian(
                sn_coord[i][2], sn_coord[i][3], 4, 32, 128)
            temp_g_img_right = tool.get_gaussian(
                sn_coord[i][4], sn_coord[i][5], 4, 32, 128)
            # convert np 2 tensor
            temp_g_img_left = torch.FloatTensor(
                np.expand_dims(temp_g_img_left, 0)).cuda()
            temp_g_img = torch.FloatTensor(
                np.expand_dims(temp_g_img, 0)).cuda()
            temp_g_img_right = torch.FloatTensor(
                np.expand_dims(temp_g_img_right, 0)).cuda()
            # B, H, W, C-> B, C, H, W
            temp_g_img_left = temp_g_img_left.permute(0, 3, 1, 2)
            temp_g_img = temp_g_img.permute(0, 3, 1, 2)
            temp_g_img_right = temp_g_img_right.permute(0, 3, 1, 2)
            # add into gaus list
            gn_img = torch.cat(
                (temp_g_img_left, temp_g_img, temp_g_img_right), dim=1)
            if(i == 0):
                # stacked image
                sn_g_img = gn_img
            else:
                sn_g_img = torch.cat((sn_g_img, gn_img), dim=0)
        sn_g_img = torch.cat((sn_img, sn_g_img), dim=1)"""

        # cropped image
        for i in range(B):
            # 1.cropped
            temp_crop_img_left = tool.cropped(
                sn_batch_np[i][0], sn_coord[i][0], sn_coord[i][1], 64, 128)
            temp_crop_img_left = torch.FloatTensor(
                np.expand_dims(temp_crop_img_left, 0)).cuda()
            temp_crop_img = tool.cropped(
                sn_batch_np[i][0], sn_coord[i][2], sn_coord[i][3], 64, 128)
            temp_crop_img = torch.FloatTensor(
                np.expand_dims(temp_crop_img, 0)).cuda()
            temp_crop_img_right = tool.cropped(
                sn_batch_np[i][0], sn_coord[i][4], sn_coord[i][5], 64, 128)
            temp_crop_img_right = torch.FloatTensor(
                np.expand_dims(temp_crop_img_right, 0)).cuda()
            # 3.permute : B, H, W, C-> B, C, H, W
            temp_crop_img_left = temp_crop_img_left.permute(0, 3, 1, 2)
            temp_crop_img = temp_crop_img.permute(0, 3, 1, 2)
            temp_crop_img_right = temp_crop_img_right.permute(0, 3, 1, 2)
            # 4.concat 3 of them in channel dimension
            #temp_crop_img
            cropn_img = torch.cat((temp_crop_img_left, temp_crop_img, temp_crop_img_right), dim=1)
            # 5. concat batch of them in batch dimension
            if(i == 0):
                # stacked crop img
                sn_crop_img = cropn_img
            else:
                sn_crop_img = torch.cat((sn_crop_img, cropn_img), dim=0)

        """end_ts = torch.FloatTensor(np.array(end_batch)).to(
            device).view(self.batch_size, 1)"""

        # TD-target
        s_coord = torch.FloatTensor(s_coord.astype(float)).to(device)
        sn_coord = torch.FloatTensor(sn_coord.astype(float)).to(device)
        with torch.no_grad():
            a_next, logpi_next, _ = self.actor.sample(
                sn_img, sn_crop_img, sn_coord)
            q_next_target = self.critic_target(
                sn_img, sn_crop_img, sn_coord, a_next) - self.alpha * logpi_next
            q_target = r_ts + self.gamma * q_next_target
            # q_target = r_ts + end_ts * self.gamma * q_next_target

        # Critic loss
        q_eval = self.critic(
            s_img, s_crop_img, s_coord, a_ts)
        self.critic_loss = self.criterion(q_eval, q_target)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        a_curr, logpi_curr, _ = self.actor.sample(
            s_img, s_crop_img, s_coord)
        q_current = self.critic(
            s_img, s_crop_img, s_coord, a_curr)
        self.actor_loss = ((self.alpha*logpi_curr) - q_current).mean()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()

        # Adaptive entropy adjustment
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpi_curr +
                                             self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())

        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())
