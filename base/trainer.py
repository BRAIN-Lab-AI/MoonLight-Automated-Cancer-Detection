import torch
import time
import numpy as np
from base import BaseTrainer
from logger import TensorboardWriter
from torch.cuda.amp import autocast, GradScaler


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, use_amp=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.writer = TensorboardWriter(config.log_dir, self.logger, config=config)
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        start = time.time()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)

            if batch_idx % self.log_step == 0:
                self.logger.debug(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.data_loader.dataset)}'
                                  f' ({100. * batch_idx / len(self.data_loader):.0f}%)] Loss: {loss.item():.6f}')

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        end = time.time()

        log = {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': end - start
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.valid_data_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += data.size(0)

        avg_val_loss = total_val_loss / total_samples
        val_accuracy = total_correct / total_samples

        return {
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        }
