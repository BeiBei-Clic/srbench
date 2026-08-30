from ..OperonRegressor import complexity,model,est
from .params._operonregressor import params

# pyoperon 参数名与旧 operon.sklearn 不同
params['optimizer_iterations'] = params.pop('local_iterations')
params['max_time'] = params.pop('time_limit')
est.set_params(**params)
est.allowed_symbols = 'add,sub,mul,div,exp,log,sin,cos,sqrt,square,constant,variable'

# double the evals
est.max_evaluations = 1000000
est.generations=100000 # just large enough since we have an evaluation budget
est.max_time=8*60*60 # 8 hours
