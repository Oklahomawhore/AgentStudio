import torch
import torch.nn.functional as F
from trl.trainer.ppo_trainer import PPOTrainer
import time
from trl.core import flatten_dict, stats_to_np, stack_dicts, PPODecorators
from typing import List, Optional


class RunningMoments:
    """计算数据的滑动均值和标准差"""
    def __init__(self, accelerator=None, decay=0.99):
        self.decay = decay
        self.accelerator = accelerator
        
        self.count = torch.tensor(0.0)
        self.mean = torch.tensor(0.0)
        self.var = torch.tensor(1.0)
        self.std = torch.tensor(1.0)
        
        if accelerator is not None:
            self.count = self.count.to(accelerator.device)
            self.mean = self.mean.to(accelerator.device)
            self.var = self.var.to(accelerator.device)
            self.std = self.std.to(accelerator.device)
    
    def update(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.mean.device)
        
        x = x.to(self.mean.device)
        
        batch_count = x.shape[0] if x.dim() > 0 else 1
        batch_mean = x.mean()
        batch_var = ((x - batch_mean) ** 2).mean()
        
        self.count += batch_count
        
        if self.count == batch_count:
            self.mean = batch_mean
            self.var = batch_var
        else:
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * (1 - self.decay)
            
            self.var = self.var * self.decay + batch_var * (1 - self.decay) + \
                      self.decay * (1 - self.decay) * (delta ** 2)
            
            self.mean = new_mean
        
        self.std = torch.sqrt(self.var + 1e-8)
        
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            self.mean = self.accelerator.gather(self.mean).mean()
            self.std = self.accelerator.gather(self.std).mean()
        
        return self.mean, self.std


class VLMPPO(PPOTrainer):
    """视觉语言模型PPO训练器，基于RL4VLM实现"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 奖励标准化器
        self.reward_normalizer = RunningMoments(self.accelerator)
        
        # 增加熵权重和KL权重设置
        self.entropy_weight = kwargs.get("entropy_weight", 0.01)
        self.kl_weight = kwargs.get("kl_weight", 0.02)
        
        # 设置PPO的优势标准化开关
        self.normalize_advantage = kwargs.get("normalize_advantage", True)
        
        # 设置学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=kwargs.get("num_train_epochs", 1000),
            eta_min=1e-6
        )

    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        """计算奖励和KL惩罚"""
        rewards = []
        kls = []
        
        # 对每个样本计算奖励和KL惩罚
        for score, mask, logprob, ref_logprob in zip(scores, masks, logprobs, ref_logprobs):
            # 创建奖励和KL惩罚张量
            reward = torch.zeros_like(mask, dtype=torch.float)
            kl = torch.zeros_like(mask, dtype=torch.float)
            
            # 找到最后一个非掩码位置
            valid_positions = mask.nonzero()
            if len(valid_positions) > 0:
                last_pos = valid_positions[-1]
                # 奖励仅分配给最后一个位置
                reward[last_pos] = score
            
            # 对所有有效位置计算KL散度
            valid_mask = mask.bool()
            if valid_mask.any():
                # 按照RL4VLM的方式，对所有token计算KL散度
                kl[valid_mask] = logprob[valid_mask] - ref_logprob[valid_mask]
            
            rewards.append(reward)
            kls.append(kl)
            
        return torch.stack(rewards), torch.zeros_like(torch.stack(rewards)), torch.stack(kls)

    def calculate_ppo_loss(self, old_logprobs, logprobs, advantages, clip_eps=0.2):
        """计算PPO损失，与RL4VLM类似"""
        # 计算概率比率
        ratio = torch.exp(logprobs - old_logprobs)
        
        # 计算裁剪和非裁剪目标
        clip_advantages = advantages.clone()
        
        # 获取正/负优势的掩码
        pos_advantages_mask = clip_advantages >= 0
        neg_advantages_mask = ~pos_advantages_mask
        
        # 对于正优势和负优势分别应用不同的裁剪策略
        # 正优势：限制上限，防止过大更新
        # 负优势：限制下限，防止过大更新
        clip_advantages[pos_advantages_mask] *= torch.min(
            torch.ones_like(ratio[pos_advantages_mask]),
            torch.clamp(ratio[pos_advantages_mask], max=1.0 + clip_eps)
        )
        
        clip_advantages[neg_advantages_mask] *= torch.max(
            torch.ones_like(ratio[neg_advantages_mask]),
            torch.clamp(ratio[neg_advantages_mask], min=1.0 - clip_eps)
        )
        
        # 计算策略损失
        policy_loss = -torch.min(ratio * advantages, clip_advantages).mean()
        
        return policy_loss, ratio

    def train_minibatch(self, old_logprobs, old_values, logprobs, logits, values, masks, advantages, returns):
        """
        训练一个小批次数据
        
        参数:
            old_logprobs: 旧策略下的动作log概率
            old_values: 旧的状态值函数
            logprobs: 当前策略下的动作log概率
            logits: 当前策略的logits输出
            values: 当前状态值函数
            masks: 有效token的mask
            advantages: 优势函数值
            returns: 回报值
        """
        # 只对有效的token计算梯度
        mask = masks.to(dtype=torch.bool)
        
        # 对优势进行标准化
        if self.normalize_advantage and mask.any():
            advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
        
        # 计算PPO策略损失
        pg_loss, ratio = self.calculate_ppo_loss(
            old_logprobs[mask], 
            logprobs[mask], 
            advantages[mask], 
            clip_eps=0.2
        )
        
        # 计算值函数损失
        v_loss = 0.5 * ((values - returns)[mask] ** 2).mean()
        
        # 计算熵损失 (基于RL4VLM实现)
        entropy = None
        if logits is not None and self.entropy_weight > 0:
            # 计算token级别的熵
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy_per_token = -(probs * log_probs).sum(-1)
            entropy = entropy_per_token[mask].mean()
        else:
            entropy = torch.tensor(0.0, device=self.current_device)
        
        # 总损失
        loss = pg_loss + 0.5 * v_loss - self.entropy_weight * entropy
        
        # 梯度清零、反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        # 记录统计信息
        stats = {
            "policy/loss": pg_loss.detach().cpu(),
            "policy/advantages": advantages[mask].detach().mean().cpu(),
            "policy/ratio": ratio.mean().cpu(),
            "policy/entropy": entropy.detach().cpu() if isinstance(entropy, torch.Tensor) else entropy,
            "value/loss": v_loss.detach().cpu(),
            "value/values": values[mask].detach().mean().cpu(),
            "value/returns": returns[mask].detach().mean().cpu(),
            "loss/total": loss.detach().cpu(),
        }
        
        return stats

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """执行一步PPO训练"""
        # 安全检查和预处理
        bs = self.config.batch_size
        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )

        scores = torch.tensor(scores, device=self.current_device)
        
        # 奖励缩放和标准化
        if self.config.use_score_scaling:
            scores_mean, scores_std = self.reward_normalizer.update(scores)
            scores = (scores - scores_mean) / (scores_std + 1e-8)
            
        # 奖励截断
        if self.config.score_clip is not None:
            scores = torch.clip(scores, -self.config.score_clip, self.config.score_clip)

        timing = dict()
        t0 = time.time()

        # 前向传播阶段 (评估模式)
        with torch.no_grad():
            t = time.time()
            
            # 处理长序列
            self.model.gradient_checkpointing_enable()
            max_length = 4096
            for i in range(len(queries)):
                total_len = queries[i].shape[0] + responses[i].shape[0]
                if total_len > max_length:
                    r_len = responses[i].shape[0]
                    q_len = min(queries[i].shape[0], max_length - r_len)
                    queries[i] = queries[i][-q_len:]
            
            # 准备模型输入
            model_inputs = self.prepare_model_inputs(queries, responses)
            
            # 处理分布式环境
            if self.is_distributed:
                pad_first = self.tokenizer.padding_side == "left"
                model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["input_ids"], dim=1,
                    pad_index=self.tokenizer.pad_token_id, pad_first=pad_first,
                )
                model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first,
                )
                if self.is_encoder_decoder:
                    # 处理encoder-decoder模型
                    pass
                    
            model_inputs_names = list(model_inputs.keys())

            # 执行前向传播
            with torch.amp.autocast(device_type='cuda', enabled=True):
                all_logprobs, logits, values, masks = self.batched_forward_pass(
                    self.model,
                    queries,
                    responses,
                    model_inputs,
                    response_masks=response_masks,
                    return_logits=True,  # 需要logits用于计算熵
                )

                # 获取参考模型的logprobs
                with self.optional_peft_ctx():
                    ref_logprobs, _, _, _ = self.batched_forward_pass(
                        self.model if self.is_peft_model else self.ref_model,
                        queries,
                        responses,
                        model_inputs,
                        return_logits=False,
                    )
                
            timing["time/ppo/forward_pass"] = time.time() - t

            # 计算奖励和KL散度
            t = time.time()
            rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            
            # 应用KL惩罚
            if self.kl_weight > 0:
                rewards = rewards - self.kl_weight * kls
                
            timing["time/ppo/compute_rewards"] = time.time() - t

            # 计算优势和回报
            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # 创建训练批次
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        # PPO训练循环
        t = time.time()
        all_stats = []
        
        # 按照RL4VLM方式进行多次迭代训练
        for epoch_idx in range(self.config.ppo_epochs):
            # 训练数据乱序
            b_inds = torch.randperm(bs)
            
            # 按批次处理
            for mini_batch_start in range(0, bs, self.config.mini_batch_size):
                mini_batch_end = mini_batch_start + self.config.mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                
                if len(mini_batch_inds) == 0:
                    continue

                # 准备小批次
                mini_batch_dict = {
                    "logprobs": batch_dict["logprobs"][mini_batch_inds],
                    "values": batch_dict["values"][mini_batch_inds],
                    "masks": batch_dict["masks"][mini_batch_inds],
                    "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                    "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                    "advantages": batch_dict["advantages"][mini_batch_inds],
                    "returns": batch_dict["returns"][mini_batch_inds],
                }
                
                # 添加模型输入
                for k in model_inputs_names:
                    mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
            
                # 训练一个小批次
                with self.accelerator.accumulate(self.model):
                    model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}
                    
                    # 使用混合精度训练
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        # 确保梯度检查点已启用
                        self.model.gradient_checkpointing_enable()
                        
                        # 执行前向传播以获取当前策略下的logprobs和values
                        logprobs, logits, vpreds, current_masks = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        
                        # 确保尺寸匹配
                        if vpreds.shape != mini_batch_dict["values"].shape:
                            if vpreds.shape[1] > mini_batch_dict["values"].shape[1]:
                                padding = torch.zeros(
                                    (mini_batch_dict["values"].shape[0], 
                                     vpreds.shape[1] - mini_batch_dict["values"].shape[1]), 
                                    device=mini_batch_dict["values"].device
                                )
                                mini_batch_dict["values"] = torch.cat([mini_batch_dict["values"], padding], dim=1)
                            else:
                                mini_batch_dict["values"] = mini_batch_dict["values"][:, :vpreds.shape[1]]
                        
                        # 调整其他张量尺寸
                        if current_masks.shape != mini_batch_dict["masks"].shape:
                            if current_masks.shape[1] > mini_batch_dict["masks"].shape[1]:
                                padding = torch.zeros(
                                    (mini_batch_dict["masks"].shape[0], 
                                     current_masks.shape[1] - mini_batch_dict["masks"].shape[1]), 
                                    device=mini_batch_dict["masks"].device
                                )
                                mini_batch_dict["masks"] = torch.cat([mini_batch_dict["masks"], padding], dim=1)
                            else:
                                mini_batch_dict["masks"] = mini_batch_dict["masks"][:, :current_masks.shape[1]]
                        
                        # 调整优势和回报尺寸
                        if mini_batch_dict["advantages"].shape != mini_batch_dict["values"].shape:
                            if mini_batch_dict["advantages"].shape[1] > mini_batch_dict["values"].shape[1]:
                                mini_batch_dict["advantages"] = mini_batch_dict["advantages"][:, :mini_batch_dict["values"].shape[1]]
                                mini_batch_dict["returns"] = mini_batch_dict["returns"][:, :mini_batch_dict["values"].shape[1]]
                            else:
                                padding = torch.zeros(
                                    (mini_batch_dict["advantages"].shape[0], 
                                     mini_batch_dict["values"].shape[1] - mini_batch_dict["advantages"].shape[1]), 
                                    device=mini_batch_dict["advantages"].device
                                )
                                mini_batch_dict["advantages"] = torch.cat([mini_batch_dict["advantages"], padding], dim=1)
                                mini_batch_dict["returns"] = torch.cat([mini_batch_dict["returns"], padding], dim=1)
                        
                        # 使用PPO进行训练
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        
                    # 清理缓存
                    del logprobs, logits, vpreds, current_masks
                    torch.cuda.empty_cache()
                    
                    all_stats.append(train_stats)

        # 处理统计信息
        t = time.time()
        train_stats = stack_dicts(all_stats)
        
        # 记录PPO统计信息
        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_weight,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )

        # 处理分布式统计数据
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # 总时间记录
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # 学习率调度
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats