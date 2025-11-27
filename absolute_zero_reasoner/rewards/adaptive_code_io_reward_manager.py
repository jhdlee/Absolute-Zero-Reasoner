"""
Adaptive Code I/O Reward Manager for Uncertainty-Guided Verification

This module extends the CodeIORewardManager to support three verification modes:
1. FULL_EXECUTION: Always use Python executor (ground truth)
2. FULL_LLM: Always use LLM-as-a-judge
3. ADAPTIVE: Use uncertainty (sc_entropy_normalized) to decide, with a budget constraint

The adaptive mode uses uncertainty quantification to determine when the LLM
verification is likely to be unreliable, and falls back to execution.
"""

import os
import time
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter
from enum import Enum
import uuid

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem

from absolute_zero_reasoner.rewards.reward_managers import CodeIORewardManager
from absolute_zero_reasoner.rewards.custom_evaluate import extract_answer
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter


class VerificationMode(Enum):
    """Verification strategy modes."""
    FULL_EXECUTION = "full_execution"
    FULL_LLM = "full_llm"
    ADAPTIVE = "adaptive"


class AdaptiveCodeIORewardManager(CodeIORewardManager):
    """
    Extended reward manager that supports different verification strategies.

    Inherits from CodeIORewardManager and adds:
    - LLM-as-a-judge verification
    - Adaptive verification using uncertainty quantification
    - Tracking of verification statistics
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int,
        split: str,
        reward_fn_extraction_type: str,
        math_metric: str,
        splitter: str,
        output_path: str,
        generation_reward_config: Dict[str, Any],
        debug: bool = False,
        max_prompt_length: int = 8192,
        valid_program_filter: str = 'all',
        batched_estimate: bool = False,
        extract_code_block: bool = True,
        num_inputs: int = 10,
        code_f_reward_type: str = 'accuracy',
        boxed_retry: bool = False,
        # New adaptive verification params
        verification_mode: str = "full_execution",
        llm_for_verification = None,
        budget_fraction: float = 0.3,
        n_samples_for_uq: int = 8,
    ):
        """
        Args:
            ... (all base class params)
            verification_mode: "full_execution", "full_llm", or "adaptive"
            llm_for_verification: vLLM instance for LLM verification (required for full_llm/adaptive)
            budget_fraction: Fraction of high-uncertainty tasks to verify with execution in adaptive mode
            n_samples_for_uq: Number of samples for uncertainty quantification
        """
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            split=split,
            reward_fn_extraction_type=reward_fn_extraction_type,
            math_metric=math_metric,
            splitter=splitter,
            output_path=output_path,
            generation_reward_config=generation_reward_config,
            debug=debug,
            max_prompt_length=max_prompt_length,
            valid_program_filter=valid_program_filter,
            batched_estimate=batched_estimate,
            extract_code_block=extract_code_block,
            num_inputs=num_inputs,
            code_f_reward_type=code_f_reward_type,
            boxed_retry=boxed_retry,
        )

        self.verification_mode = VerificationMode(verification_mode)
        self.llm_for_verification = llm_for_verification  # Can be None, will use rollout_actor_wg
        self.budget_fraction = budget_fraction
        self.n_samples_for_uq = n_samples_for_uq

        # Tracking metrics
        self.verification_stats = {
            "total_verifications": 0,
            "execution_verifications": 0,
            "llm_verifications": 0,
            "execution_time_ms": 0.0,
            "llm_time_ms": 0.0,
            "agreements": 0,
            "disagreements": 0,
        }

        # Track per-step metrics for logging
        self.step_stats = []

    def reset_stats(self):
        """Reset verification statistics."""
        for key in self.verification_stats:
            self.verification_stats[key] = 0 if isinstance(self.verification_stats[key], int) else 0.0

    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """
        Extended __call__ that supports different verification modes.

        For generation tasks (gen_*), uses base class implementation.
        For prediction tasks (pred_*), applies the configured verification mode.
        """
        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # For generation tasks, use base class (they require execution anyway)
        if problem_type is not None and problem_type.startswith('gen'):
            return super().__call__(
                data=data,
                problem_type=problem_type,
                executor=executor,
                rollout_actor_wg=rollout_actor_wg,
                banned_words=banned_words,
                banned_assertion_keywords=banned_assertion_keywords,
                n_samples=n_samples,
                input_type_counters=input_type_counters,
                output_type_counters=output_type_counters,
                error_type_counters=error_type_counters,
            )

        # For prediction tasks, apply verification mode
        if self.verification_mode == VerificationMode.FULL_EXECUTION:
            return super().__call__(
                data=data,
                problem_type=problem_type,
                executor=executor,
                rollout_actor_wg=rollout_actor_wg,
                banned_words=banned_words,
                banned_assertion_keywords=banned_assertion_keywords,
                n_samples=n_samples,
                input_type_counters=input_type_counters,
                output_type_counters=output_type_counters,
                error_type_counters=error_type_counters,
            )
        elif self.verification_mode == VerificationMode.FULL_LLM:
            return self._call_with_llm_verification(
                data=data,
                problem_type=problem_type,
                executor=executor,
                banned_words=banned_words,
                banned_assertion_keywords=banned_assertion_keywords,
                rollout_actor_wg=rollout_actor_wg,
            )
        elif self.verification_mode == VerificationMode.ADAPTIVE:
            return self._call_with_adaptive_verification(
                data=data,
                problem_type=problem_type,
                executor=executor,
                rollout_actor_wg=rollout_actor_wg,
                banned_words=banned_words,
                banned_assertion_keywords=banned_assertion_keywords,
                n_samples=n_samples,
            )
        else:
            raise ValueError(f"Unknown verification mode: {self.verification_mode}")

    def _call_with_llm_verification(
        self,
        data: DataProto,
        problem_type: str,
        executor,
        banned_words: List[str],
        banned_assertion_keywords: List[str],
        rollout_actor_wg=None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """
        Compute rewards using LLM-as-a-judge verification for prediction tasks.
        Uses the rollout_actor_wg (the solver model itself) for verification.
        """
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = defaultdict(list)
        data_dicts = []
        correct_predictions = []

        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
        if problem_type is None:
            problem_types = [d.non_tensor_batch['extra_info']['metric'] for d in data]
            problem_type = 'pred'
        else:
            problem_types = [problem_type] * len(data)

        PrettyPrinter.section_header("Getting Data Dicts for LLM Verification")
        for i in range(len(data)):
            data_dict = self._get_data_dict(
                data[i], problem_types[i], executor, banned_words, uids[i], banned_assertion_keywords
            )
            data_dicts.append(data_dict)

        # Construct LLM verification prompts
        PrettyPrinter.section_header("LLM Verification for Prediction Tasks")
        verifier_prompts_data = []
        valid_indices = []

        for i, data_dict in enumerate(data_dicts):
            if not data_dict['format_score']:
                continue

            answer = data_dict.get('answer')
            if answer is None:
                continue

            prompt = self._construct_verifier_prompt(
                data_dict=data_dict,
                problem_type=problem_types[i],
            )
            verifier_prompts_data.append({
                'prompt': [{'role': 'user', 'content': prompt}],
                'uid': data_dict['uid'],
                'data_source': data_dict['data_source'],
                'extra_info': data_dict['extra_info'],
                'ground_truth': '',
            })
            valid_indices.append(i)

        # Batch LLM verification using rollout_actor_wg
        llm_verdicts = {}
        if verifier_prompts_data and rollout_actor_wg is not None:
            start_time = time.time()

            # Create temporary dataset for verification prompts
            import os
            temp_path = f'{self.output_path}/temp_verifier.parquet'
            pd.DataFrame(verifier_prompts_data).to_parquet(temp_path)

            temp_data = RLHFDataset(
                parquet_files=temp_path,
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            os.remove(temp_path)

            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)
            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )

            batch_data = next(iter(dataloader))
            gen_batch = DataProto.from_single_dict(batch_data)
            gen_batch = gen_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,  # Greedy for verification
                'validate': False,
            }

            # Generate
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

            llm_time = (time.time() - start_time) * 1000

            # Extract verdicts
            for out_idx in range(len(output_gen_batch)):
                response = self.tokenizer.decode(
                    output_gen_batch[out_idx].batch['responses'],
                    skip_special_tokens=True
                )
                task_idx = valid_indices[out_idx]
                verdict = self._extract_llm_verdict(response)
                llm_verdicts[task_idx] = {
                    "verdict": verdict,
                    "response": response,
                }

            self.verification_stats["llm_verifications"] += len(verifier_prompts_data)
            self.verification_stats["llm_time_ms"] += llm_time

        # Compute rewards based on LLM verdicts
        acc_rewards = []
        for i, data_dict in enumerate(data_dicts):
            valid_response_length = data_dict['valid_response_length']

            if not data_dict['format_score']:
                acc_reward = 0.0
                reward_tensor[i, valid_response_length - 1] = -1.0
            elif i in llm_verdicts:
                acc_reward = 1.0 if llm_verdicts[i]["verdict"] else 0.0
                if self.split == 'train':
                    if acc_reward > 0:
                        reward_tensor[i, valid_response_length - 1] = acc_reward
                    else:
                        reward_tensor[i, valid_response_length - 1] = -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = acc_reward

                if acc_reward > 0:
                    correct_predictions.append(data_dict)
            else:
                acc_reward = 0.0
                reward_tensor[i, valid_response_length - 1] = -0.5

            acc_rewards.append(acc_reward)

        all_scores['accuracy'] = acc_rewards
        all_scores['format_score'] = [d['format_score'] for d in data_dicts]
        all_scores['verification_method'] = ['llm'] * len(data_dicts)

        self.verification_stats["total_verifications"] += len(data)

        # Add verification metrics to all_scores for logging
        all_scores['verification_execution_count'] = [0]
        all_scores['verification_llm_count'] = [len(verifier_prompts_data)]
        all_scores['verification_execution_fraction'] = [0.0]
        all_scores['verification_llm_fraction'] = [1.0]

        return reward_tensor, all_scores, [], correct_predictions

    def _call_with_adaptive_verification(
        self,
        data: DataProto,
        problem_type: str,
        executor,
        rollout_actor_wg,
        banned_words: List[str],
        banned_assertion_keywords: List[str],
        n_samples: int,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """
        Compute rewards using adaptive verification based on uncertainty.

        Strategy:
        1. Get data dicts with format checks
        2. For valid predictions, compute uncertainty using self-consistency
        3. Sort by uncertainty
        4. Use execution for top budget_fraction of high-uncertainty tasks
        5. Use LLM for the rest
        """
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = defaultdict(list)
        data_dicts = []
        correct_predictions = []

        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
        if problem_type is None:
            problem_types = [d.non_tensor_batch['extra_info']['metric'] for d in data]
            problem_type = 'pred'
        else:
            problem_types = [problem_type] * len(data)

        PrettyPrinter.section_header("Getting Data Dicts for Adaptive Verification")
        for i in range(len(data)):
            data_dict = self._get_data_dict(
                data[i], problem_types[i], executor, banned_words, uids[i], banned_assertion_keywords
            )
            data_dicts.append(data_dict)

        # Compute uncertainties using self-consistency sampling
        PrettyPrinter.section_header("Computing Uncertainties via Self-Consistency")
        uncertainties = self._compute_uncertainties_batch(data_dicts, problem_types, rollout_actor_wg)

        for i, unc in enumerate(uncertainties):
            data_dicts[i]['uncertainty'] = unc

        # Get valid indices (those with format_score > 0)
        valid_indices = [i for i, d in enumerate(data_dicts) if d['format_score'] > 0]

        if not valid_indices:
            # All invalid, return early
            all_scores['accuracy'] = [0.0] * len(data_dicts)
            all_scores['format_score'] = [d['format_score'] for d in data_dicts]
            all_scores['verification_method'] = ['none'] * len(data_dicts)
            all_scores['uncertainty'] = uncertainties
            return reward_tensor, all_scores, [], []

        # Sort valid indices by uncertainty (descending)
        sorted_valid_indices = sorted(
            valid_indices,
            key=lambda i: data_dicts[i].get('uncertainty', 0),
            reverse=True
        )

        # Determine which tasks use execution vs LLM
        n_execution = int(len(sorted_valid_indices) * self.budget_fraction)
        execution_indices = set(sorted_valid_indices[:n_execution])
        llm_indices = set(sorted_valid_indices[n_execution:])

        # Verify with execution (high uncertainty)
        PrettyPrinter.section_header(f"Execution Verification ({len(execution_indices)} tasks)")
        exec_rewards = {}
        for i in execution_indices:
            start_time = time.time()

            data_dict = data_dicts[i]
            answer = data_dict.get('answer')
            imports = data_dict.get('imports', [])

            if answer is None:
                acc_reward = 0.0
            elif problem_types[i].endswith('code_i'):
                acc_reward = executor.eval_input_prediction(
                    code=data_dict['program'],
                    gold_output=data_dict['output'],
                    agent_input=answer,
                    imports=list(set(imports))
                )
                acc_reward = acc_reward if acc_reward is not None else 0.0
            elif problem_types[i].endswith('code_o'):
                acc_reward = executor.eval_output_prediction(
                    code=data_dict['program'],
                    gold_output=data_dict['output'],
                    agent_output=answer,
                    imports=list(set(imports))
                )
                acc_reward = acc_reward if acc_reward is not None else 0.0
            else:
                acc_reward = 0.0

            exec_time = (time.time() - start_time) * 1000
            exec_rewards[i] = acc_reward

            self.verification_stats["execution_verifications"] += 1
            self.verification_stats["execution_time_ms"] += exec_time

        # Verify with LLM using rollout_actor_wg (low uncertainty)
        PrettyPrinter.section_header(f"LLM Verification ({len(llm_indices)} tasks)")
        llm_rewards = {}

        if llm_indices and rollout_actor_wg is not None:
            verifier_prompts_data = []
            prompt_to_idx = []

            for i in llm_indices:
                data_dict = data_dicts[i]
                if data_dict.get('answer') is not None:
                    prompt = self._construct_verifier_prompt(data_dict, problem_types[i])
                    verifier_prompts_data.append({
                        'prompt': [{'role': 'user', 'content': prompt}],
                        'uid': data_dict['uid'],
                        'data_source': data_dict['data_source'],
                        'extra_info': data_dict['extra_info'],
                        'ground_truth': '',
                    })
                    prompt_to_idx.append(i)

            if verifier_prompts_data:
                start_time = time.time()

                # Create temporary dataset for verification prompts
                import os
                temp_path = f'{self.output_path}/temp_verifier_adaptive.parquet'
                pd.DataFrame(verifier_prompts_data).to_parquet(temp_path)

                temp_data = RLHFDataset(
                    parquet_files=temp_path,
                    tokenizer=self.tokenizer,
                    prompt_key='prompt',
                    max_prompt_length=self.max_prompt_length,
                    filter_prompts=True,
                    return_raw_chat=False,
                    truncation='error'
                )
                os.remove(temp_path)

                sampler = torch.utils.data.SequentialSampler(data_source=temp_data)
                dataloader = torch.utils.data.DataLoader(
                    dataset=temp_data,
                    batch_size=len(temp_data),
                    drop_last=False,
                    shuffle=False,
                    collate_fn=collate_fn,
                    sampler=sampler,
                )

                batch_data = next(iter(dataloader))
                gen_batch = DataProto.from_single_dict(batch_data)
                gen_batch = gen_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,  # Greedy for verification
                    'validate': False,
                }

                # Generate
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
                output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
                output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

                llm_time = (time.time() - start_time) * 1000

                # Extract verdicts
                for out_idx in range(len(output_gen_batch)):
                    response = self.tokenizer.decode(
                        output_gen_batch[out_idx].batch['responses'],
                        skip_special_tokens=True
                    )
                    task_idx = prompt_to_idx[out_idx]
                    verdict = self._extract_llm_verdict(response)
                    llm_rewards[task_idx] = 1.0 if verdict else 0.0

                self.verification_stats["llm_verifications"] += len(verifier_prompts_data)
                self.verification_stats["llm_time_ms"] += llm_time

        # Combine rewards and compute final tensor
        acc_rewards = []
        verification_methods = []

        for i, data_dict in enumerate(data_dicts):
            valid_response_length = data_dict['valid_response_length']

            if not data_dict['format_score']:
                acc_reward = 0.0
                method = 'none'
                reward_tensor[i, valid_response_length - 1] = -1.0
            elif i in exec_rewards:
                acc_reward = exec_rewards[i]
                method = 'execution'
                if self.split == 'train':
                    if acc_reward > 0:
                        reward_tensor[i, valid_response_length - 1] = acc_reward
                    else:
                        reward_tensor[i, valid_response_length - 1] = -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = acc_reward
            elif i in llm_rewards:
                acc_reward = llm_rewards[i]
                method = 'llm'
                if self.split == 'train':
                    if acc_reward > 0:
                        reward_tensor[i, valid_response_length - 1] = acc_reward
                    else:
                        reward_tensor[i, valid_response_length - 1] = -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = acc_reward
            else:
                acc_reward = 0.0
                method = 'none'
                reward_tensor[i, valid_response_length - 1] = -0.5

            if acc_reward > 0:
                correct_predictions.append(data_dict)

            acc_rewards.append(acc_reward)
            verification_methods.append(method)

        all_scores['accuracy'] = acc_rewards
        all_scores['format_score'] = [d['format_score'] for d in data_dicts]
        all_scores['verification_method'] = verification_methods
        all_scores['uncertainty'] = uncertainties

        self.verification_stats["total_verifications"] += len(data)

        # Log step stats
        n_valid = len(valid_indices)
        step_stat = {
            'total': len(data),
            'execution_count': len(execution_indices),
            'llm_count': len(llm_indices),
            'mean_uncertainty': np.mean(uncertainties) if uncertainties else 0.0,
            'mean_reward': np.mean(acc_rewards),
        }
        self.step_stats.append(step_stat)

        # Add verification metrics to all_scores for logging
        all_scores['verification_execution_count'] = [len(execution_indices)]
        all_scores['verification_llm_count'] = [len(llm_indices)]
        all_scores['verification_execution_fraction'] = [len(execution_indices) / n_valid if n_valid > 0 else 0.0]
        all_scores['verification_llm_fraction'] = [len(llm_indices) / n_valid if n_valid > 0 else 0.0]
        all_scores['mean_uncertainty'] = [np.mean(uncertainties) if uncertainties else 0.0]

        return reward_tensor, all_scores, [], correct_predictions

    def _compute_uncertainties_batch(
        self,
        data_dicts: List[Dict],
        problem_types: List[str],
        rollout_actor_wg,
    ) -> List[float]:
        """
        Compute uncertainties for all tasks using self-consistency.

        This uses the rollout actor to generate multiple responses and
        computes the normalized entropy of the answer distribution.
        """
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset
        from absolute_zero_reasoner.data_construction.constructor import get_code_problem_predictor_prompt
        from absolute_zero_reasoner.data_construction.process_data import instruction_following

        if self.n_samples_for_uq <= 1 or rollout_actor_wg is None:
            # No UQ possible, return default uncertainty
            return [0.5] * len(data_dicts)

        # Prepare prompts for self-consistency sampling
        # We need to re-generate solver responses with temperature > 0
        uq_prompts_data = []
        valid_indices = []

        for i, data_dict in enumerate(data_dicts):
            if not data_dict['format_score']:
                continue

            # Construct solver prompt for this task
            program = data_dict.get('program', '')
            gold_input = data_dict.get('input', '')
            gold_output = data_dict.get('output', '')

            if problem_types[i].endswith('code_i'):
                predictor_prompt = get_code_problem_predictor_prompt(
                    problem_type='code_i',
                    snippet=program,
                    input_args=gold_input,
                    output=gold_output,
                )
            elif problem_types[i].endswith('code_o'):
                predictor_prompt = get_code_problem_predictor_prompt(
                    problem_type='code_o',
                    snippet=program,
                    input_args=gold_input,
                    output=gold_output,
                )
            else:
                continue

            prompt = instruction_following.format(predictor_prompt)

            uq_prompts_data.append({
                'prompt': [{'role': 'user', 'content': prompt}],
                'uid': data_dict['uid'],
                'data_source': data_dict['data_source'],
                'extra_info': data_dict['extra_info'],
                'ground_truth': '',
            })
            valid_indices.append(i)

        # Initialize all uncertainties to 0.5 (default)
        uncertainties = [0.5] * len(data_dicts)

        if not uq_prompts_data:
            return uncertainties

        # Sample multiple responses
        # Repeat the prompts n_samples times
        uq_prompts_data_repeated = uq_prompts_data * self.n_samples_for_uq

        try:
            import os
            temp_path = f'{self.output_path}/temp_uq.parquet'
            pd.DataFrame(uq_prompts_data_repeated).to_parquet(temp_path)

            temp_data = RLHFDataset(
                parquet_files=temp_path,
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            os.remove(temp_path)

            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)
            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )

            batch_data = next(iter(dataloader))
            gen_batch = DataProto.from_single_dict(batch_data)
            gen_batch = gen_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,  # Enable sampling for diversity
                'validate': False,
            }

            # Generate
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

            # Extract answers and group by task
            answers_by_task = {i: [] for i in valid_indices}
            n_prompts = len(uq_prompts_data)

            for out_idx in range(len(output_gen_batch)):
                # Map back to original task
                task_pos = out_idx % n_prompts
                task_idx = valid_indices[task_pos]

                response = self.tokenizer.decode(
                    output_gen_batch[out_idx].batch['responses'],
                    skip_special_tokens=True
                )

                # Extract answer from response
                extracted = extract_answer(response, self.reward_fn_extraction_type, boxed_retry=self.boxed_retry)
                if problem_types[task_idx].endswith('code_i'):
                    answer = self.extract_input_output(extracted, return_input=True, return_output=False)
                else:
                    answer = self.extract_input_output(extracted, return_input=False, return_output=True)

                answers_by_task[task_idx].append(str(answer) if answer else '')

            # Compute normalized entropy for each task
            for task_idx, answers in answers_by_task.items():
                if not answers:
                    uncertainties[task_idx] = 1.0
                    continue

                # Compute normalized self-consistency entropy
                entropy = self._compute_normalized_entropy(answers)
                uncertainties[task_idx] = entropy

        except Exception as e:
            PrettyPrinter.print_colored(f"UQ sampling failed: {e}", "red")
            # Fall back to default uncertainty
            pass

        return uncertainties

    def _compute_normalized_entropy(self, answers: List[str]) -> float:
        """Compute normalized entropy of answer distribution."""
        n = len(answers)
        if n == 0:
            return 1.0

        # Count answer frequencies
        counts = Counter(answers)

        # Compute entropy
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max entropy (log2(n))
        max_entropy = math.log2(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def _construct_verifier_prompt(
        self,
        data_dict: Dict,
        problem_type: str,
    ) -> str:
        """Construct LLM verifier prompt for a prediction task."""
        program = data_dict.get('program', '')
        answer = data_dict.get('answer', '')
        gold_output = data_dict.get('output', '')
        gold_input = data_dict.get('input', '')

        if problem_type.endswith('code_o'):
            return f"""You are a code verification assistant. Determine if the predicted output is correct.

Code:
```python
{program}
```

Input: {gold_input}
Expected Output: {gold_output}
Predicted Output: {answer}

Is the predicted output correct? Consider that outputs may be formatted differently but represent the same value.
Answer with only "CORRECT" or "INCORRECT"."""

        elif problem_type.endswith('code_i'):
            return f"""You are a code verification assistant. Determine if the predicted input would produce the expected output.

Code:
```python
{program}
```

Expected Output: {gold_output}
Predicted Input: {answer}

Would this input produce the expected output when run through the code?
Answer with only "CORRECT" or "INCORRECT"."""

        return ""

    def _extract_llm_verdict(self, response: str) -> bool:
        """Extract verdict from LLM verifier response."""
        response_upper = response.upper().strip()

        # Check for explicit keywords
        if "INCORRECT" in response_upper:
            return False
        if "CORRECT" in response_upper:
            return True

        # Fallback checks
        if response_upper.startswith("YES"):
            return True
        if response_upper.startswith("NO"):
            return False

        # Default to incorrect if unclear
        return False

    def get_wandb_metrics(self) -> Dict:
        """Get metrics formatted for wandb logging."""
        stats = self.get_verification_stats()

        metrics = {
            "verification/total_verifications": stats.get("total_verifications", 0),
            "verification/execution_count": stats.get("execution_verifications", 0),
            "verification/llm_count": stats.get("llm_verifications", 0),
            "verification/execution_fraction": stats.get("execution_fraction", 0),
            "verification/llm_fraction": stats.get("llm_fraction", 0),
            "verification/avg_execution_time_ms": stats.get("avg_execution_time_ms", 0),
            "verification/avg_llm_time_ms": stats.get("avg_llm_time_ms", 0),
        }

        # Add step-level stats if available
        if self.step_stats:
            latest_step = self.step_stats[-1]
            metrics["verification/step_mean_uncertainty"] = latest_step.get("mean_uncertainty", 0)
            metrics["verification/step_mean_reward"] = latest_step.get("mean_reward", 0)

        return metrics

    def get_verification_stats(self) -> Dict:
        """Get summary of verification statistics."""
        total = self.verification_stats["total_verifications"]
        if total == 0:
            return self.verification_stats

        summary = {
            **self.verification_stats,
            "execution_fraction": self.verification_stats["execution_verifications"] / total,
            "llm_fraction": self.verification_stats["llm_verifications"] / total,
            "avg_execution_time_ms": (
                self.verification_stats["execution_time_ms"] / self.verification_stats["execution_verifications"]
                if self.verification_stats["execution_verifications"] > 0 else 0
            ),
            "avg_llm_time_ms": (
                self.verification_stats["llm_time_ms"] / self.verification_stats["llm_verifications"]
                if self.verification_stats["llm_verifications"] > 0 else 0
            ),
        }

        return summary
