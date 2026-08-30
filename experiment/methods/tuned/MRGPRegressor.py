from ..MRGPRegressor import complexity, model, est
from ..src.mrgp import MRGPRegressor
from .params._mrgpregressor import params

est.set_params(**params)

# 官方协议在此加倍预算(50万评估)，但本机 50 万评估约 83 分钟，
# 超过 evaluate_model 对小数据集的 3600 秒硬限，超时强杀后无模型输出。
# 故保留官方调参最优预算(25万评估，约 42 分钟)，并用 MRGP 软超时
# 兜底：50 分钟到时优雅停机，输出当前最优模型。
est.time_out=50*60
