"""
Adaptive Code I/O Reward Manager for Uncertainty-Guided Verification

This module extends the CodeIORewardManager to support three verification modes:
1. FULL_EXECUTION: Always use Python executor (ground truth)
2. FULL_LLM: Always use LLM reflection with majority voting (no execution)
3. ADAPTIVE: Use reflection vote entropy to route uncertain tasks to execution

The key idea:
- For each task, the solver outputs ONE answer
- The solver is prompted N times with a reflection prompt asking "Is this answer correct?"
- Majority voting of reflections determines correctness (Full LLM mode)
- Entropy of reflection votes measures uncertainty (Adaptive mode)
- High entropy (mixed votes) → uncertain → use execution
- Low entropy (consistent votes) → confident → trust majority vote
"""

import os
import time
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from enum import Enum
import uuid

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem

from absolute_zero_reasoner.rewards.reward_managers import CodeIORewardManager
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
    - LLM reflection-based verification with majority voting
    - Adaptive verification using reflection vote entropy
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
        # Adaptive verification params
        verification_mode: str = "full_execution",
        llm_for_verification = None,
        budget_fraction: float = 0.3,
        n_samples_for_uq: int = 8,
    ):
        """
        Args:
            ... (all base class params)
            verification_mode: "full_execution", "full_llm", or "adaptive"
            llm_for_verification: Not used (kept for API compatibility)
            budget_fraction: Fraction of high-uncertainty tasks to verify with execution in adaptive mode
            n_samples_for_uq: Number of reflection samples for uncertainty quantification
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
        self.llm_for_verification = llm_for_verification  # Kept for API compatibility
        self.budget_fraction = budget_fraction
        self.n_samples_for_uq = n_samples_for_uq

        # Tracking metrics
        self.verification_stats = {
            "total_verifications": 0,
            "execution_verifications": 0,
            "llm_verifications": 0,
            "execution_time_ms": 0.0,
            "llm_time_ms": 0.0,
        }

        # Track per-step metrics for logging
        self.step_stats = []

        # Debug counters
        self._debug_log_count = 0
        self._debug_vote_count = 0

    def reset_stats(self):
        """Reset verification statistics."""
        for key in self.verification_stats:
            self.verification_stats[key] = 0 if isinstance(self.verification_stats[key], int) else 0.0
        self._debug_log_count = 0
        self._debug_vote_count = 0

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
        Compute rewards using LLM reflection with majority voting.

        For each task:
        1. Solver has already produced ONE answer
        2. Prompt N times asking "Is this answer correct?"
        3. Majority vote of reflections determines correctness
        """
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

        print(f"\n{'='*60}")
        print(f"[DEBUG LLM VERIFICATION] Starting with {len(data)} tasks")
        print(f"  problem_type: {problem_type}")
        print(f"  n_samples_for_uq: {self.n_samples_for_uq}")
        print(f"{'='*60}")

        PrettyPrinter.section_header("Getting Data Dicts for LLM Verification")
        for i in range(len(data)):
            data_dict = self._get_data_dict(
                data[i], problem_types[i], executor, banned_words, uids[i], banned_assertion_keywords
            )
            data_dicts.append(data_dict)

        # Debug: analyze data_dicts
        n_valid_format = sum(1 for d in data_dicts if d.get('format_score', False))
        n_has_answer = sum(1 for d in data_dicts if d.get('answer') is not None)
        print(f"\n[DEBUG DATA DICTS]")
        print(f"  Total tasks: {len(data_dicts)}")
        print(f"  Valid format_score: {n_valid_format}")
        print(f"  Has answer: {n_has_answer}")

        # Show first few data_dicts
        for i, d in enumerate(data_dicts[:3]):
            print(f"\n  [Task {i}]")
            print(f"    format_score: {d.get('format_score')}")
            print(f"    answer: {str(d.get('answer'))[:100] if d.get('answer') else 'None'}...")
            print(f"    program: {str(d.get('program'))[:80] if d.get('program') else 'None'}...")
            print(f"    input: {str(d.get('input'))[:50] if d.get('input') else 'None'}")
            print(f"    output: {str(d.get('output'))[:50] if d.get('output') else 'None'}")

        # Get reflection votes for all valid tasks
        PrettyPrinter.section_header(f"LLM Reflection Verification (N={self.n_samples_for_uq} reflections per task)")

        reflection_results = self._get_reflection_votes_batch(
            data_dicts=data_dicts,
            problem_types=problem_types,
            rollout_actor_wg=rollout_actor_wg,
        )

        print(f"\n[DEBUG REFLECTION RESULTS]")
        print(f"  Tasks with reflection results: {len(reflection_results)}")
        if reflection_results:
            # Show vote distribution for first few tasks
            for task_idx, result in list(reflection_results.items())[:5]:
                votes = result.get('votes', [])
                responses = result.get('responses', [])
                print(f"\n  [Task {task_idx}]")
                print(f"    Votes: {votes}")
                print(f"    CORRECT count: {sum(votes)}/{len(votes)}")
                print(f"    Majority: {'CORRECT' if sum(votes) > len(votes)/2 else 'INCORRECT'}")
                if responses:
                    print(f"    First response (truncated): {responses[0][:200]}...")

        # Compute rewards based on majority voting
        acc_rewards = []
        for i, data_dict in enumerate(data_dicts):
            valid_response_length = data_dict['valid_response_length']

            if not data_dict['format_score']:
                acc_reward = 0.0
                reward_tensor[i, valid_response_length - 1] = -1.0
            elif i in reflection_results:
                votes = reflection_results[i]['votes']
                correct_count = sum(votes)
                total_votes = len(votes)

                # Majority voting
                majority_correct = correct_count > total_votes / 2
                acc_reward = 1.0 if majority_correct else 0.0

                if self.split == 'train':
                    reward_tensor[i, valid_response_length - 1] = acc_reward if acc_reward > 0 else -0.5
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

        self.verification_stats["total_verifications"] += len(data)
        self.verification_stats["llm_verifications"] += len(reflection_results)

        # Add verification metrics
        all_scores['verification_execution_count'] = [0]
        all_scores['verification_llm_count'] = [len(reflection_results)]
        all_scores['verification_execution_fraction'] = [0.0]
        all_scores['verification_llm_fraction'] = [1.0]

        # Debug: final summary
        print(f"\n[DEBUG FINAL SUMMARY]")
        print(f"  Total tasks: {len(data_dicts)}")
        print(f"  Tasks with reflection results: {len(reflection_results)}")
        print(f"  Accuracy rewards: {acc_rewards[:10]}... (first 10)")
        print(f"  Mean accuracy: {np.mean(acc_rewards):.4f}")
        print(f"  Correct predictions: {len(correct_predictions)}")
        print(f"{'='*60}\n")

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
        Compute rewards using adaptive verification based on reflection vote entropy.

        Strategy:
        1. For each task, get N reflection votes
        2. Compute entropy of votes (high entropy = uncertain)
        3. Route high-entropy tasks to execution verification
        4. Route low-entropy tasks to majority vote from reflections
        """
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

        # Get reflection votes and compute entropy for all valid tasks
        PrettyPrinter.section_header(f"Computing Reflection Votes (N={self.n_samples_for_uq})")

        reflection_results = self._get_reflection_votes_batch(
            data_dicts=data_dicts,
            problem_types=problem_types,
            rollout_actor_wg=rollout_actor_wg,
        )

        # Compute entropies and store in data_dicts
        uncertainties = []
        valid_indices = []
        for i, data_dict in enumerate(data_dicts):
            if i in reflection_results:
                votes = reflection_results[i]['votes']
                entropy = self._compute_vote_entropy(votes)
                data_dict['uncertainty'] = entropy
                data_dict['reflection_votes'] = votes
                uncertainties.append(entropy)
                valid_indices.append(i)
            else:
                data_dict['uncertainty'] = 1.0  # Max uncertainty for invalid
                uncertainties.append(1.0)

        if not valid_indices:
            # All invalid, return early
            all_scores['accuracy'] = [0.0] * len(data_dicts)
            all_scores['format_score'] = [d['format_score'] for d in data_dicts]
            all_scores['uncertainty'] = uncertainties
            return reward_tensor, all_scores, [], []

        # Sort valid indices by uncertainty (descending)
        sorted_valid_indices = sorted(
            valid_indices,
            key=lambda i: data_dicts[i].get('uncertainty', 0),
            reverse=True
        )

        # Determine which tasks use execution vs LLM majority vote
        n_execution = int(len(sorted_valid_indices) * self.budget_fraction)
        execution_indices = set(sorted_valid_indices[:n_execution])
        llm_indices = set(sorted_valid_indices[n_execution:])

        PrettyPrinter.section_header(f"Execution Verification ({len(execution_indices)} high-uncertainty tasks)")

        # Verify with execution (high uncertainty)
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

        PrettyPrinter.section_header(f"LLM Majority Vote ({len(llm_indices)} low-uncertainty tasks)")

        # Use majority vote from reflections (low uncertainty)
        llm_rewards = {}
        for i in llm_indices:
            if i in reflection_results:
                votes = reflection_results[i]['votes']
                correct_count = sum(votes)
                majority_correct = correct_count > len(votes) / 2
                llm_rewards[i] = 1.0 if majority_correct else 0.0
                self.verification_stats["llm_verifications"] += 1

        # Combine rewards and compute final tensor
        acc_rewards = []
        for i, data_dict in enumerate(data_dicts):
            valid_response_length = data_dict['valid_response_length']

            if not data_dict['format_score']:
                acc_reward = 0.0
                reward_tensor[i, valid_response_length - 1] = -1.0
            elif i in exec_rewards:
                acc_reward = exec_rewards[i]
                if self.split == 'train':
                    reward_tensor[i, valid_response_length - 1] = acc_reward if acc_reward > 0 else -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = acc_reward
            elif i in llm_rewards:
                acc_reward = llm_rewards[i]
                if self.split == 'train':
                    reward_tensor[i, valid_response_length - 1] = acc_reward if acc_reward > 0 else -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = acc_reward
            else:
                acc_reward = 0.0
                reward_tensor[i, valid_response_length - 1] = -0.5

            if acc_reward > 0:
                correct_predictions.append(data_dict)

            acc_rewards.append(acc_reward)

        all_scores['accuracy'] = acc_rewards
        all_scores['format_score'] = [d['format_score'] for d in data_dicts]
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

        # Add verification metrics
        all_scores['verification_execution_count'] = [len(execution_indices)]
        all_scores['verification_llm_count'] = [len(llm_indices)]
        all_scores['verification_execution_fraction'] = [len(execution_indices) / n_valid if n_valid > 0 else 0.0]
        all_scores['verification_llm_fraction'] = [len(llm_indices) / n_valid if n_valid > 0 else 0.0]
        all_scores['mean_uncertainty'] = [np.mean(uncertainties) if uncertainties else 0.0]

        return reward_tensor, all_scores, [], correct_predictions

    def _get_reflection_votes_batch(
        self,
        data_dicts: List[Dict],
        problem_types: List[str],
        rollout_actor_wg,
    ) -> Dict[int, Dict]:
        """
        Get N reflection votes for each task.

        For each valid task:
        1. Construct reflection prompt: "Is [answer] correct for [task]?"
        2. Generate N responses with temperature > 0
        3. Extract CORRECT/INCORRECT votes from each response

        Returns:
            Dict mapping task index to {'votes': [bool, ...], 'responses': [...]}
        """
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset

        if rollout_actor_wg is None:
            print("[DEBUG] rollout_actor_wg is None! Cannot generate reflections.")
            print("  This is required for LLM-based verification.")
            return {}

        # Prepare reflection prompts for valid tasks
        reflection_prompts = []
        valid_indices = []
        skipped_format = 0
        skipped_answer = 0

        for i, data_dict in enumerate(data_dicts):
            if not data_dict['format_score']:
                skipped_format += 1
                continue

            answer = data_dict.get('answer')
            if answer is None:
                skipped_answer += 1
                continue

            prompt = self._construct_reflection_prompt(
                data_dict=data_dict,
                problem_type=problem_types[i],
            )

            # Skip empty prompts (e.g., unsupported problem types)
            if not prompt or not prompt.strip():
                print(f"[DEBUG] Skipping task {i}: empty reflection prompt for {problem_types[i]}")
                continue

            # Debug: log first few prompts
            if self._debug_log_count < 3:
                print(f"[DEBUG Reflection Prompt {self._debug_log_count}]")
                print(f"  problem_type: {problem_types[i]}")
                print(f"  answer: {str(answer)[:100]}...")
                print(f"  prompt preview: {prompt[:300]}...")
                self._debug_log_count += 1

            reflection_prompts.append({
                'prompt': [{'role': 'user', 'content': prompt}],
                'uid': data_dict['uid'],
                'data_source': data_dict['data_source'],
                'extra_info': data_dict['extra_info'],
                'ground_truth': '',
            })
            valid_indices.append(i)

        if not reflection_prompts:
            print(f"[DEBUG] No reflection prompts to generate")
            print(f"  Skipped due to format_score=False: {skipped_format}")
            print(f"  Skipped due to answer=None: {skipped_answer}")
            return {}

        # Repeat prompts N times for sampling
        prompts_repeated = reflection_prompts * self.n_samples_for_uq

        print(f"\n[DEBUG REFLECTION GENERATION]")
        print(f"  Total tasks: {len(data_dicts)}")
        print(f"  Skipped (format_score=False): {skipped_format}")
        print(f"  Skipped (answer=None): {skipped_answer}")
        print(f"  Valid tasks for reflection: {len(reflection_prompts)}")
        print(f"  Total prompts (N={self.n_samples_for_uq} samples): {len(prompts_repeated)}")

        try:
            start_time = time.time()

            temp_path = f'{self.output_path}/temp_reflection.parquet'
            pd.DataFrame(prompts_repeated).to_parquet(temp_path)

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
                'do_sample': True,  # Enable sampling for diversity in reflections
                'validate': False,
            }

            # Generate
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

            llm_time = (time.time() - start_time) * 1000
            self.verification_stats["llm_time_ms"] += llm_time

            # Extract votes and group by task
            results = {i: {'votes': [], 'responses': []} for i in valid_indices}
            n_prompts = len(reflection_prompts)

            for out_idx in range(len(output_gen_batch)):
                # Map back to original task
                task_pos = out_idx % n_prompts
                task_idx = valid_indices[task_pos]

                response = self.tokenizer.decode(
                    output_gen_batch[out_idx].batch['responses'],
                    skip_special_tokens=True
                )

                vote = self._extract_reflection_vote(response)
                results[task_idx]['votes'].append(vote)
                results[task_idx]['responses'].append(response)

            # Debug: log first few results
            debug_count = 0
            for task_idx, result in results.items():
                if debug_count < 2:
                    votes = result['votes']
                    print(f"[DEBUG Reflection Result] Task {task_idx}")
                    print(f"  Votes: {votes} ({sum(votes)}/{len(votes)} CORRECT)")
                    print(f"  Entropy: {self._compute_vote_entropy(votes):.3f}")
                    print(f"  Sample response: {result['responses'][0][:150]}...")
                    debug_count += 1

            return results

        except Exception as e:
            print(f"\n[DEBUG ERROR] Reflection sampling failed!")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            PrettyPrinter.status("ERROR", f"Reflection sampling failed: {e}", "error")
            import traceback
            traceback.print_exc()
            return {}

    def _construct_reflection_prompt(
        self,
        data_dict: Dict,
        problem_type: str,
    ) -> str:
        """
        Construct reflection prompt asking if the solver's answer is correct.

        The prompt shows the task and the solver's answer, then asks for verification.
        """
        program = data_dict.get('program', '')
        answer = data_dict.get('answer', '')
        gold_output = data_dict.get('output', '')
        gold_input = data_dict.get('input', '')

        if problem_type.endswith('code_o'):
            return f"""Given the following Python code and input, determine if the predicted output is correct.

Code:
```python
{program}
```

Input: {gold_input}
Predicted Output: {answer}

Think step by step about what the code does with this input, then answer:
Is the predicted output correct? Answer with CORRECT or INCORRECT."""

        elif problem_type.endswith('code_i'):
            return f"""Given the following Python code and expected output, determine if the predicted input would produce that output.

Code:
```python
{program}
```

Expected Output: {gold_output}
Predicted Input: {answer}

Think step by step about what input would produce the expected output, then answer:
Is the predicted input correct? Answer with CORRECT or INCORRECT."""

        elif problem_type.endswith('code_f'):
            # For pred_code_f: given_inputs/given_outputs are shown to the model
            given_inputs = data_dict.get('given_inputs', [])
            given_outputs = data_dict.get('given_outputs', [])
            message = data_dict.get('message', '')

            # Format given I/O pairs (what the model saw)
            io_pairs = []
            for i, (inp, out) in enumerate(zip(given_inputs[:3], given_outputs[:3])):
                io_pairs.append(f"  Input {i+1}: {inp}\n  Output {i+1}: {out}")
            io_str = "\n".join(io_pairs) if io_pairs else "  (no examples provided)"

            return f"""Given a function completion task, determine if the predicted implementation is correct.

Task Description: {message}

Predicted Function:
```python
{answer}
```

Example Input/Output pairs:
{io_str}

Think step by step about whether the predicted function would produce the correct outputs for the given inputs, then answer:
Is the predicted function correct? Answer with CORRECT or INCORRECT."""

        return ""

    def _extract_reflection_vote(self, response: str) -> bool:
        """
        Extract CORRECT/INCORRECT vote from reflection response.

        Returns True for CORRECT, False for INCORRECT.
        """
        response_upper = response.upper().strip()

        # Check for explicit keywords (order matters - check negative first)
        negative_patterns = [
            "INCORRECT", "NOT CORRECT", "WRONG", "NOT RIGHT",
            "FALSE", "NO,", "NO.", "NO ", "WOULDN'T", "WOULD NOT",
        ]
        positive_patterns = [
            "CORRECT", "RIGHT", "TRUE", "YES,", "YES.", "YES ",
            "WOULD PRODUCE", "MATCHES", "EQUAL",
        ]

        # Debug: log extraction for first few responses
        if self._debug_vote_count < 10:
            matched_neg = [p for p in negative_patterns if p in response_upper]
            matched_pos = [p for p in positive_patterns if p in response_upper]
            print(f"[DEBUG VOTE EXTRACTION {self._debug_vote_count}]")
            print(f"  Response (first 150 chars): {response[:150]}...")
            print(f"  Matched negative: {matched_neg}")
            print(f"  Matched positive: {matched_pos}")
            self._debug_vote_count += 1

        # Check negative patterns first
        for pattern in negative_patterns:
            if pattern in response_upper:
                return False

        # Then check positive patterns
        for pattern in positive_patterns:
            if pattern in response_upper:
                return True

        # Default to incorrect if unclear
        if self._debug_vote_count < 15:
            print(f"  -> Defaulting to INCORRECT (no patterns matched)")
            self._debug_vote_count += 1

        return False

    def _compute_vote_entropy(self, votes: List[bool]) -> float:
        """
        Compute normalized entropy of reflection votes.

        For binary votes: H = -p*log2(p) - (1-p)*log2(1-p)
        Normalized by max entropy (1.0 for binary)

        Returns:
            0.0 = all votes agree (low uncertainty)
            1.0 = votes are split 50/50 (high uncertainty)
        """
        if not votes:
            return 1.0

        n = len(votes)
        correct_count = sum(votes)

        # Edge cases: all same vote
        if correct_count == 0 or correct_count == n:
            return 0.0

        # Binary entropy
        p = correct_count / n
        entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)

        # Already normalized for binary (max entropy = 1.0)
        return entropy

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
