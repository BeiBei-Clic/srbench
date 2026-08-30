from ..ITEARegressor import complexity,model,eval_kwargs, est

# itea 1.0.0 参数名 npop/ngens 已改为 popsize/gens，
# nonzeroexps/transfunctions 已移除（tfuncs 在 est 构造中指定）
# 官方最优参数 npop=1000, ngens=500 即 est 默认的 popsize/gens

# double the evals
est.popsize = int(est.popsize*2**0.5)
est.gens = int(est.gens*2**0.5)
