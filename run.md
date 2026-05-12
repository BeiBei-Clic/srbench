# srbench 算法批量测试

## 运行命令

```bash
cd /home/xyh/Symbolic_Regression/srbench/experiment
python3 run_experience_test.py
```

## 说明

- 数据集：`1089_USCrime`（47行），`test=True` 小参数模式
- 测试 15 个算法：AdaBoostRegressor、KernelRidge、MLPRegressor、RandomForestRegressor、gplearn、OperonRegressor、ITEARegressor、MRGPRegressor、LGBMRegressor、XGBRegressor、ffx、gpgomea、tuned.AFPRegressor、tuned.DSRRegressor、tuned.EHCRegressor
- 结果输出到 `results/experience_test/`，每个算法一个 JSON 文件
- 结尾打印汇总表（状态、R²、MSE、MAE、耗时）
