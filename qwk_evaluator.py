"""
QWK (Quadratic Weighted Kappa) Evaluator
计算 Trait-level 和 Overall QWK，按 prompt 分组计算后取平均
"""

from typing import Dict, List, Any
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
import numpy as np


class QWKEvaluator:
    """QWK 评估器：按 prompt 分组计算 QWK 后取平均"""
    
    def __init__(self, min_samples_per_prompt: int = 2):
        """
        Args:
            min_samples_per_prompt: 每个 prompt 最少样本数，少于此数跳过
        """
        self.min_samples_per_prompt = min_samples_per_prompt
    
    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算 Trait-level 和 Overall QWK
        
        Args:
            samples: 样本列表，每个样本包含：
                - prompt_id: int
                - gold_trait_scores: Dict[str, int/float]  # trait 名称 -> 金标准分数
                - pred_trait_scores: Dict[str, int/float]  # trait 名称 -> 预测分数
                - gold_overall_score: int/float
                - pred_overall_score: int/float
        
        Returns:
            {
                "trait_qwk": {"trait1": float, "trait2": float, ...},
                "overall_qwk": float
            }
        """
        # Step 1: 按 prompt_id 分组
        grouped = self._group_by_prompt(samples)
        
        # Step 2: 计算 Trait-level QWK
        trait_qwk = self._compute_trait_qwk(grouped)
        
        # Step 3: 计算 Overall QWK
        overall_qwk = self._compute_overall_qwk(grouped)
        
        return {
            "trait_qwk": trait_qwk,
            "overall_qwk": overall_qwk
        }
    
    def _group_by_prompt(self, samples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """按 prompt_id 分组"""
        grouped = defaultdict(list)
        for sample in samples:
            prompt_id = sample["prompt_id"]
            grouped[prompt_id].append(sample)
        return dict(grouped)
    
    def _compute_trait_qwk(self, grouped: Dict[int, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        计算 Trait-level QWK
        对每个 trait，先按 prompt 计算 QWK，然后取平均
        """
        # 收集所有 trait 名称
        trait_names = set()
        for samples in grouped.values():
            for sample in samples:
                if "gold_trait_scores" in sample:
                    trait_names.update(sample["gold_trait_scores"].keys())
        
        trait_qwk_results = {}
        
        for trait in trait_names:
            prompt_qwks = []
            
            # 对每个 prompt 计算该 trait 的 QWK
            for prompt_id, samples in grouped.items():
                y_true = []
                y_pred = []
                
                for sample in samples:
                    gold_scores = sample.get("gold_trait_scores", {})
                    pred_scores = sample.get("pred_trait_scores", {})
                    
                    if trait in gold_scores and trait in pred_scores:
                        y_true.append(gold_scores[trait])
                        y_pred.append(pred_scores[trait])
                
                # 样本数足够才计算
                if len(y_true) >= self.min_samples_per_prompt:
                    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                    prompt_qwks.append(qwk)
            
            # 对该 trait 的所有 prompt QWK 取平均
            if prompt_qwks:
                trait_qwk_results[trait] = float(np.mean(prompt_qwks))
            else:
                trait_qwk_results[trait] = 0.0
        
        return trait_qwk_results
    
    def _compute_overall_qwk(self, grouped: Dict[int, List[Dict[str, Any]]]) -> float:
        """
        计算 Overall QWK
        先按 prompt 计算 QWK，然后取平均
        """
        prompt_qwks = []
        
        for prompt_id, samples in grouped.items():
            y_true = []
            y_pred = []
            
            for sample in samples:
                if "gold_overall_score" in sample and "pred_overall_score" in sample:
                    y_true.append(sample["gold_overall_score"])
                    y_pred.append(sample["pred_overall_score"])
            
            # 样本数足够才计算
            if len(y_true) >= self.min_samples_per_prompt:
                qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                prompt_qwks.append(qwk)
        
        # 对所有 prompt QWK 取平均
        if prompt_qwks:
            return float(np.mean(prompt_qwks))
        else:
            return 0.0


def compute_qwk_metrics(samples: List[Dict[str, Any]], min_samples_per_prompt: int = 2) -> Dict[str, Any]:
    """
    便捷函数：计算 QWK 指标
    
    Args:
        samples: 样本列表
        min_samples_per_prompt: 每个 prompt 最少样本数
    
    Returns:
        {
            "trait_qwk": {"trait1": float, ...},
            "overall_qwk": float
        }
    """
    evaluator = QWKEvaluator(min_samples_per_prompt=min_samples_per_prompt)
    return evaluator.evaluate(samples)
