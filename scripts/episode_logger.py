import jax
import jax.numpy as jp
import numpy as np
from typing import Dict, Any

class FinalDistanceLogger:
    """专门记录episode结束时的最终距离"""
    
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.final_distances = []
        self.success_count = 0
        self.total_episodes = 0

    def update(self, metrics: Dict[str, Any], dones: jax.Array):  # 改这里
        """更新指标，只记录episode结束时的数据"""
        if jp.sum(dones) > 0:
            # 获取结束的episode的最终距离
            final_dist = metrics.get("final_distance_to_goal", jp.zeros_like(dones))
            episode_success = metrics.get("episode_success", jp.zeros_like(dones))
            
            # 只取done=True的episode的数据
            done_mask = dones.astype(bool)
            if jp.any(done_mask):
                final_distances_batch = final_dist[done_mask]
                success_batch = episode_success[done_mask]
                
                # 添加到缓冲区
                self.final_distances.extend(final_distances_batch.tolist())
                self.success_count += jp.sum(success_batch).item()
                self.total_episodes += jp.sum(done_mask).item()
                
                # 保持缓冲区大小
                if len(self.final_distances) > self.buffer_size:
                    excess = len(self.final_distances) - self.buffer_size
                    self.final_distances = self.final_distances[excess:]
    
    def get_summary(self) -> Dict[str, float]:
        """获取统计摘要"""
        if not self.final_distances:
            return {
                "mean_final_distance": 0.0,
                "std_final_distance": 0.0,
                "success_rate": 0.0,
                "min_final_distance": 0.0,
                "max_final_distance": 0.0,
                "episodes_recorded": 0
            }
        
        distances = np.array(self.final_distances)
        return {
            "mean_final_distance": float(np.mean(distances)),
            "std_final_distance": float(np.std(distances)),
            "success_rate": float(self.success_count / max(self.total_episodes, 1)),
            "min_final_distance": float(np.min(distances)),
            "max_final_distance": float(np.max(distances)),
            "episodes_recorded": len(self.final_distances)
        }
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        print("=" * 50)
        print("FINAL DISTANCE TO GOAL SUMMARY")
        print("=" * 50)
        print(f"Episodes recorded: {summary['episodes_recorded']}")
        print(f"Mean final distance: {summary['mean_final_distance']:.3f} m")
        print(f"Std final distance: {summary['std_final_distance']:.3f} m")
        print(f"Min final distance: {summary['min_final_distance']:.3f} m")
        print(f"Max final distance: {summary['max_final_distance']:.3f} m")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print("=" * 50)