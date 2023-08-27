import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self._basedir = './saved_models'

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # 1: predicted Q values with current state
        pred = self.model(state) # (num_states, num_actions)
        target = pred.clone() # (num_states, num_actions)
        
        for state_id in range(len(done)):
            if not done[state_id]:
                Q_new = reward[state_id] + self.gamma * torch.max(self.model(next_state[state_id]))
            else:
                Q_new = reward[state_id]

            target[state_id][action[state_id]] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
    def _make_basedir(self):
        if not os.path.exists(self._basedir):
            os.makedirs(self._basedir)
            
    def _get_fullpath(self, name, suffix):
        return os.path.join(self._basedir, f'{name}_{suffix}')
        
    def save(self, name, suffix):
        self._make_basedir()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'gamma': self.gamma
        }, self._get_fullpath(name, suffix))
        
    def load(self, name, suffix):
        checkpoint = torch.load(self._get_fullpath(name, suffix))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']