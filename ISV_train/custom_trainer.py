import torch
from trl.trainer.ppo_trainer import PPOTrainer
import time
from trl.core import flatten_dict, stats_to_np, stack_dicts, PPODecorators
from typing import List, Optional


class RunningMoments:
    """
    计算数据的滑动均值和标准差
    """
    def __init__(self, accelerator=None, decay=0.99):
        """
        初始化滑动矩估计器
        
        参数:
            accelerator: Accelerator对象，用于分布式训练
            decay: 滑动平均的衰减率，越大表示历史数据影响越大
        """
        self.decay = decay
        self.accelerator = accelerator
        
        # 初始化均值和不中心二阶矩
        self.count = torch.tensor(0.0)
        self.mean = torch.tensor(0.0)
        self.var = torch.tensor(1.0)
        self.std = torch.tensor(1.0)
        
        # 使用加速器设备
        if accelerator is not None:
            self.count = self.count.to(accelerator.device)
            self.mean = self.mean.to(accelerator.device)
            self.var = self.var.to(accelerator.device)
            self.std = self.std.to(accelerator.device)
    
    def update(self, x):
        """
        更新统计数据并返回均值和标准差
        
        参数:
            x: 输入数据，可以是张量或标量
            
        返回:
            tuple: (均值, 标准差)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.mean.device)
        
        # 确保输入数据与均值在同一设备上
        x = x.to(self.mean.device)
        
        batch_count = x.shape[0] if x.dim() > 0 else 1
        batch_mean = x.mean()
        batch_var = ((x - batch_mean) ** 2).mean()
        
        # 更新计数器
        self.count += batch_count
        
        # 如果是首次更新
        if self.count == batch_count:
            self.mean = batch_mean
            self.var = batch_var
        else:
            # Welford在线更新算法的变体，带有衰减
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * (1 - self.decay)
            
            # 更新方差
            # 新方差 = 旧方差*衰减 + 批次方差*(1-衰减) + 衰减*(1-衰减)*(旧均值-批次均值)^2
            self.var = self.var * self.decay + batch_var * (1 - self.decay) + \
                      self.decay * (1 - self.decay) * (delta ** 2)
            
            self.mean = new_mean
        
        # 重新计算标准差
        self.std = torch.sqrt(self.var + 1e-8)  # 添加小值避免数值不稳定
        
        # 在分布式环境中同步统计数据
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            # 同步均值和标准差
            self.mean = self.accelerator.gather(self.mean).mean()
            self.std = self.accelerator.gather(self.std).mean()
        
        return self.mean, self.std


class CustomPPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 创建奖励的滑动平均估计器
        self.reward_running = RunningMoments(self.accelerator)

    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        """
        重写计算奖励的方法，去掉KL惩罚
        """
        rewards = []
        # 对于每个样本，只关注分数作为奖励
        for score, mask in zip(scores, masks):
            reward = torch.zeros_like(mask, dtype=torch.float)
            last_non_masked_index = mask.nonzero()[-1]

            # 将奖励分配到最后一个非掩码位置
            reward[last_non_masked_index] += score
            rewards.append(reward)

        # 返回奖励、空的非分数奖励和空的KL值
        return torch.stack(rewards), torch.zeros_like(torch.stack(rewards)), torch.zeros_like(torch.stack(rewards))

    def compute_advantages(self, values, rewards, mask):
        """
        重写计算优势的方法，使用奖励的滑动平均作为基线
        """
        # 更新奖励的滑动平均
        flattened_rewards = rewards.flatten()[mask.flatten() > 0]
        reward_mean, reward_std = self.reward_running.update(flattened_rewards)

        # 使用奖励的滑动平均作为基线
        advantages = rewards - reward_mean

        if self.config.whiten_rewards:
            advantages = advantages / (reward_std + 1e-8)

        # 返回原始值、计算的优势和回报
        returns = advantages + values
        advantages = advantages.detach()

        return values, advantages, returns

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        重写PPO优化步骤，简化KL惩罚计算并使用奖励滑动平均
        增加内存优化策略处理超长序列
        """
        bs = self.config.batch_size
        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )

        scores = torch.tensor(scores, device=self.current_device)

        # 对分数进行缩放
        if self.config.use_score_scaling:
            scores_mean, scores_std = self.reward_running.update(scores)
            score_scaling_factor = scores_std + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - scores_mean) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        # 分数截断
        if self.config.score_clip is not None:
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        timing = dict()
        t0 = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        # 分布式环境的处理
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"
            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        # 前向传播计算logprobs和values
        with torch.no_grad():
            t = time.time()
            # 1. 启用梯度检查点以减少内存使用
            self.model.gradient_checkpointing_enable()
            
            # 2. 使用低精度进行前向传播计算
            with torch.amp.autocast(enabled=True):
                all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                    self.model,
                    queries,
                    responses,
                    model_inputs,
                    response_masks=response_masks,
                    return_logits=False,
                )

                # 获取参考模型的logprobs以进行记录，但不用于计算奖励
                with self.optional_peft_ctx():
                    ref_logprobs, _, _, _ = self.batched_forward_pass(
                        self.model if self.is_peft_model else self.ref_model,
                        queries,
                        responses,
                        model_inputs,
                        return_logits=False,
                    )

            # 3. 明确释放不再需要的大型中间变量
            if 'logits_or_none' in locals() and logits_or_none is not None:
                del logits_or_none
                torch.cuda.empty_cache()
                
            timing["time/ppo/forward_pass"] = time.time() - t

            # 计算奖励和优势
            t = time.time()
            rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

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
        early_stop = False
        for epoch_idx in range(self.config.ppo_epochs):
            if early_stop:
                break

            b_inds = torch.randperm(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, len(backward_batch_inds), self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    
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
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]

                    # 4. 检查序列长度，如果太长，则截断
                    max_length = 4096  # 设置一个合理的最大长度
                    trunc_queries = []
                    trunc_responses = []
                    for q, r in zip(mini_batch_dict["queries"], mini_batch_dict["responses"]):
                        # 确保query和response的总长度不超过max_length
                        total_len = q.shape[0] + r.shape[0]
                        if total_len > max_length:
                            # 保留完整的response，截断query的前部
                            r_len = r.shape[0]
                            q_len = min(q.shape[0], max_length - r_len)
                            q = q[-q_len:]  # 保留query的后部分
                        trunc_queries.append(q)
                        trunc_responses.append(r)
                    
                    mini_batch_dict["queries"] = trunc_queries
                    mini_batch_dict["responses"] = trunc_responses
                    
                    # 重新生成model_inputs以适应截断后的序列
                    trunc_model_inputs = self.prepare_model_inputs(trunc_queries, trunc_responses)
                    for k in model_inputs_names:
                        mini_batch_dict[k] = trunc_model_inputs[k]
                    
                    # 训练一个小批次
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}
                        
                        # 5. 使用混合精度训练
                        with torch.amp.autocast(enabled=True):
                            # 确保梯度检查点已启用
                            if not hasattr(self.model, "is_gradient_checkpointing") or not self.model.is_gradient_checkpointing:
                                self.model.gradient_checkpointing_enable()
                                
                            # 前向传播
                            logprobs, logits, vpreds, _ = self.batched_forward_pass(
                                self.model,
                                mini_batch_dict["queries"],
                                mini_batch_dict["responses"],
                                model_inputs,
                                return_logits=False,
                            )
                            
                            # 6. 分段计算loss以减少内存使用
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
                            
                        # 7. 在每个小批次处理后清理缓存
                        del logprobs, logits, vpreds
                        torch.cuda.empty_cache()
                        
                        all_stats.append(train_stats)

                # 8. 在每个backward批次后明确清理GPU缓存
                torch.cuda.empty_cache()

        # 记录统计信息
        t = time.time()
        train_stats = stack_dicts(all_stats)

        # 重塑优势/比率以便不平均它们
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        # 记录步骤统计信息
        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=0.0,  # KL系数设为0
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )

        # 收集/减少所有进程的统计信息
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # 记录总PPO时间
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats