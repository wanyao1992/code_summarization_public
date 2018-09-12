import datetime
import math
import os
import time

from torch.autograd import Variable
import torch

import lib
import sys

class ReinforceTrainer(object):
    def __init__(self, actor, critic, train_data, eval_data, metrics, dicts, optim, critic_optim, opt):
        self.actor = actor
        self.critic = critic

        self.train_data = train_data
        self.eval_data = eval_data
        self.evaluator = lib.Evaluator(actor, metrics, dicts, opt)

        self.actor_loss_func = metrics["xent_loss"]
        self.critic_loss_func = metrics["critic_loss"]
        self.sent_reward_func = metrics["sent_reward"]

        self.dicts = dicts

        self.optim = optim
        self.critic_optim = critic_optim

        self.max_length = opt.max_predict_length
        self.pert_func = opt.pert_func
        self.opt = opt

    def train(self, start_epoch, end_epoch, pretrain_critic, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.optim.last_loss = self.critic_optim.last_loss = None
        self.optim.set_lr(self.opt.reinforce_lr)

        #  Use large learning rate for critic during pre-training.
        if pretrain_critic:
            self.critic_optim.set_lr(1e-3)
        else:
            self.critic_optim.set_lr(self.opt.reinforce_lr)

        for epoch in range(start_epoch, end_epoch + 1):
            print("* REINFORCE epoch *")
            print("Actor optim lr: %g; Critic optim lr: %g" % (self.optim.lr, self.critic_optim.lr))
            if pretrain_critic:
                print("Pretrain critic...")
            no_update = self.opt.no_update and (not pretrain_critic) and (epoch == start_epoch)

            if no_update: print("No update...")

            train_reward, critic_loss = self.train_epoch(epoch, pretrain_critic, no_update)
            print("Train sentence reward: %.2f" % (train_reward * 100))
            print("Critic loss: %g" % critic_loss)
            # sys.exit()
            valid_loss, valid_sent_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print("Validation perplexity: %.2f" % valid_ppl)
            print("Validation sentence reward: %.2f" % (valid_sent_reward * 100))
            print("Validation corpus reward: %.2f" % (valid_corpus_reward * 100))

            if no_update: break

            self.optim.updateLearningRate(-valid_sent_reward, epoch)
            # Actor and critic use the same lr when jointly trained.
            # TODO: using small lr for critic is better?
            if not pretrain_critic:
                self.critic_optim.set_lr(self.optim.lr)

            checkpoint = {
                "model": self.actor,
                "critic": self.critic,
                "dicts": self.dicts,
                "opt": self.opt,
                "epoch": epoch,
                "optim": self.optim,
                "critic_optim": self.critic_optim
            }
            model_name = os.path.join(self.opt.save_dir, "model_rf_%s_%s_%s" % (self.opt.data_type, self.opt.has_attn, epoch))
            if pretrain_critic:
                model_name += "_pretrain"
            else:
                model_name += "_reinforce"
            model_name += ".pt"
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)

    def train_epoch(self, epoch, pretrain_critic, no_update):
        self.actor.train()
        total_reward, report_reward = 0, 0
        total_critic_loss, report_critic_loss = 0, 0
        total_sents, report_sents = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()
        batch_order = torch.randperm(len(self.train_data))
        for i in range(len(self.train_data)): #
            batch = self.train_data[i] # batch_order[i]
            if self.opt.data_type == 'code':
                targets = batch[2]
                attention_mask = batch[1][2][0].data.eq(lib.Constants.PAD).t()
            elif self.opt.data_type == 'text':
                targets = batch[2]
                attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
            elif self.opt.data_type == 'hybrid':
                targets = batch[2]
                attention_mask_code = batch[1][2][0].data.eq(lib.Constants.PAD).t()
                attention_mask_txt = batch[0][0].data.eq(lib.Constants.PAD).t()

            batch_size = targets.size(1)

            self.actor.zero_grad()
            self.critic.zero_grad()

            # Sample translations
            if self.opt.has_attn:
                if self.opt.data_type == 'code' or self.opt.data_type == 'text':
                    self.actor.decoder.attn.applyMask(attention_mask)
                elif self.opt.data_type == 'hybrid':
                    self.actor.decoder.attn.applyMask(attention_mask_code, attention_mask_txt)
            samples, outputs = self.actor.sample(batch, self.max_length)

            # Calculate rewards
            rewards, samples = self.sent_reward_func(samples.t().tolist(), targets.data.t().tolist())
            reward = sum(rewards)

            # Perturb rewards (if specified).
            if self.pert_func is not None:
                rewards = self.pert_func(rewards)

            samples = Variable(torch.LongTensor(samples).t().contiguous())
            rewards = Variable(torch.FloatTensor([rewards] * samples.size(0)).contiguous())
            if self.opt.cuda:
                samples = samples.cuda()
                rewards = rewards.cuda()

            # Update critic.
            critic_weights = samples.ne(lib.Constants.PAD).float()
            num_words = critic_weights.data.sum()
            if not no_update:
                if self.opt.data_type == 'code':
                    baselines = self.critic((batch[0], batch[1], samples, batch[3]), eval=False, regression=True)
                elif self.opt.data_type == 'text':
                    baselines = self.critic((batch[0], batch[1], samples, batch[3]), eval=False, regression=True)
                elif self.opt.data_type == 'hybrid':
                    baselines = self.critic((batch[0], batch[1], samples, batch[3]), eval=False, regression=True)

                critic_loss = self.critic.backward(baselines, rewards, critic_weights, num_words, self.critic_loss_func, regression=True)
                self.critic_optim.step()
            else:
                critic_loss = 0

            # Update actor
            if not pretrain_critic and not no_update:
                # Subtract baseline from reward
                norm_rewards = Variable((rewards - baselines).data)
                actor_weights = norm_rewards * critic_weights
                # TODO: can use PyTorch reinforce() here but that function is a black box.
                # This is an alternative way where you specify an objective that gives the same gradient
                # as the policy gradient's objective, which looks much like weighted log-likelihood.
                actor_loss = self.actor.backward(outputs, samples, actor_weights, 1, self.actor_loss_func)
                self.optim.step()
            else:
                actor_loss = 0

            # Gather stats
            total_reward += reward
            report_reward += reward
            total_sents += batch_size
            report_sents += batch_size
            total_critic_loss += critic_loss
            report_critic_loss += critic_loss
            total_words += num_words
            report_words += num_words
            self.opt.iteration += 1
            print ("iteration: %s, loss: %s " % (self.opt.iteration, actor_loss))
            print ("iteration: %s, reward: %s " % (self.opt.iteration, (report_reward / report_sents) * 100))

            if i % self.opt.log_interval == 0 and i > 0:
                print("""Epoch %3d, %6d/%d batches; actor reward: %.4f; critic loss: %f; %5.0f tokens/s; %s elapsed""" %
                      (epoch, i, len(self.train_data), (report_reward / report_sents) * 100,
                      report_critic_loss / report_words,
                      report_words / (time.time() - last_time),
                      str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                report_reward = report_sents = report_critic_loss = report_words = 0
                last_time = time.time()

        return total_reward / total_sents, total_critic_loss / total_words

