"""异常分数校准：基于 Isotonic Regression 把 raw sigmoid 分数映射到真实概率。

用法：
    # 离线拟合
    python scripts/fit_calibrator.py --config configs/default.yaml \
        --checkpoint checkpoints/best_model.pth --output checkpoints/calibrator.pkl

    # 在 configs/default.yaml 中设 calibration.enabled: true
    # 推理时 NSFWDetector 会自动加载并应用校准
"""
import os
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ScoreCalibrator:
    """Isotonic Regression 分数校准器。

    把模型的 raw sigmoid 分数（未校准）映射到真实概率。
    需要 normal(0) + abnormal(1) 两类样本拟合。
    """

    def __init__(self):
        self.iso = None
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        """拟合校准器。

        Args:
            raw_scores: [N] float in [0,1]，模型的原始异常分数
            labels: [N] int in {0,1}，0=normal, 1=abnormal
        """
        self.iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        self.iso.fit(np.asarray(raw_scores, dtype=np.float64),
                     np.asarray(labels, dtype=np.float64))
        self.fitted = True

    def transform(self, raw_score: float) -> float:
        """把 raw 分数转为校准后概率。未拟合时返回原值。"""
        if not self.fitted:
            return float(raw_score)
        return float(self.iso.transform([float(raw_score)])[0])

    def transform_batch(self, raw_scores: np.ndarray) -> np.ndarray:
        """批量校准。"""
        if not self.fitted:
            return np.asarray(raw_scores, dtype=np.float32)
        return self.iso.transform(np.asarray(raw_scores, dtype=np.float64)).astype(np.float32)

    def save(self, path: str):
        """保存到 pickle 文件。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'iso': self.iso, 'fitted': self.fitted}, f)

    @classmethod
    def load(cls, path: str) -> 'ScoreCalibrator':
        """从文件加载。文件不存在时返回未拟合的实例（向后兼容）。"""
        cal = cls()
        if not os.path.exists(path):
            cal.fitted = False
            return cal
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            cal.iso = data['iso']
            cal.fitted = data['fitted']
        except Exception:
            cal.fitted = False
        return cal
