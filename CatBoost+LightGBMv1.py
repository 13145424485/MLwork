import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor, Pool
from scipy import stats
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
import multiprocessing
from joblib import Parallel, delayed

start_time = time.time()

# 设置全局并行参数
n_jobs = multiprocessing.cpu_count() - 1
print(f"使用 {n_jobs} 个CPU核心进行并行处理")

# 列定义
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
CL = columns + noise_columns

# 基于误差分析的特征重要性权重
feature_importance_weights = {
    'T_SONIC': 2.0,            # 中高误差 (0.8268)
    'CO2_density': 2.5,         # 高误差 (1.1029)
    'CO2_density_fast_tmpr': 3.0, # 最高误差 (1.1625)
    'H2O_density': 1.5,         # 中等误差 (0.6396)
    'H2O_sig_strgth': 0.5,      # 低误差 (0.0599)
    'CO2_sig_strgth': 0.5       # 低误差 (0.0754)
}

# 优化的双流特征提取函数 - 针对高误差特征增强
def extract_dual_stream_features(df_input):
    """创建针对高误差特征优化的特征"""
    df = df_input.copy()
    
    # 基础交互特征
    df['noise_interaction'] = df['Error_CO2_density'] * df['Error_H2O_density']
    
    # 高误差特征专用交互特征
    high_error_interactions = [
        # CO2_density 相关交互
        ('Error_CO2_density', 'Error_CO2_density_fast_tmpr'),
        ('Error_CO2_density', 'Error_T_SONIC'),
        # CO2_density_fast_tmpr 相关交互
        ('Error_CO2_density_fast_tmpr', 'Error_T_SONIC'),
        ('Error_CO2_density_fast_tmpr', 'Error_H2O_density'),
        # T_SONIC 相关交互
        ('Error_T_SONIC', 'Error_H2O_density')
    ]
    
    for col1, col2 in high_error_interactions:
        if col1 in df.columns and col2 in df.columns:
            df[f'interact_{col1}_{col2}'] = df[col1] * df[col2]
    
    # 为高误差特征使用多个窗口大小
    high_error_cols = ['Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_T_SONIC']
    window_sizes = [3, 5, 10]
    
    for col in high_error_cols:
        if col in df.columns:
            for window in window_sizes:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
    
    # 为低误差特征使用单一窗口
    low_error_cols = ['Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    window_size = 5
    for col in low_error_cols:
        if col in df.columns:
            df[f'{col}_roll_mean_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).mean()
            df[f'{col}_roll_std_{window_size}'] = df[col].rolling(window=window_size, min_periods=1).std()
    
    # 为高误差特征增加滞后特征
    lags = [1, 2, 3, 5]
    for col in high_error_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                df[f'{col}_diff_{lag}'] = df[col].diff(lag)
    
    # 低误差特征只使用简单滞后
    for col in low_error_cols[:2]:  # 只为部分低误差特征创建
        if col in df.columns:
            df[f'{col}_lag_1'] = df[col].shift(1)
    
    # 特别针对CO2_density和CO2_density_fast_tmpr的关系特征
    if 'Error_CO2_density' in df.columns and 'Error_CO2_density_fast_tmpr' in df.columns:
        df['co2_density_ratio'] = df['Error_CO2_density'] / (df['Error_CO2_density_fast_tmpr'].abs() + 1e-6)
        df['co2_density_diff'] = df['Error_CO2_density'] - df['Error_CO2_density_fast_tmpr']
    
    # 填充缺失值
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

# 优化的数据预处理流程 - 针对高误差特征
def enhanced_preprocessing(df_input):
    """针对高误差特征优化的特征工程流程"""
    df = df_input.copy()
    df = extract_dual_stream_features(df)
    
    # 为高误差特征创建更多多项式特征
    high_error_cols = ['Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_T_SONIC']
    poly_features_df = pd.DataFrame(index=df.index)
    
    for col in high_error_cols:
        if col in df.columns:
            poly_features_df[f'{col}_squared'] = df[col] ** 2
            poly_features_df[f'{col}_cubed'] = df[col] ** 3  # 为高误差特征保留三次方
            # 为CO2密度相关特征添加指数变换
            if 'CO2' in col:
                poly_features_df[f'{col}_exp'] = np.exp(np.clip(df[col], -5, 5))
    
    # 为低误差特征只创建平方项
    low_error_cols = ['Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    for col in low_error_cols:
        if col in df.columns:
            poly_features_df[f'{col}_squared'] = df[col] ** 2
    
    df_out = pd.concat([df, poly_features_df], axis=1)
    
    # 填充缺失值
    numeric_cols_out = df_out.select_dtypes(include=np.number).columns
    df_out[numeric_cols_out] = df_out[numeric_cols_out].fillna(df_out[numeric_cols_out].median())
    
    return df_out

# 优化的特征选择函数 - 针对高误差特征加权
def select_important_features(X_train, y_train, X_val, X_test, threshold=0.01):
    """针对高误差特征优化的特征选择"""
    print("执行针对高误差特征的特征选择...")
    
    feature_selector = CatBoostRegressor(
        iterations=100,  # 减少迭代次数
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        random_seed=217,
        verbose=0,
        task_type='CPU',  # 明确指定任务类型
        thread_count=n_jobs  # 使用并行处理
    )
    
    # 为每个目标分别进行特征选择，但对高误差特征给予更高权重
    def get_important_features_for_target(target_col):
        # 为高误差特征使用更低的阈值
        target_threshold = threshold
        if target_col in feature_importance_weights:
            weight = feature_importance_weights[target_col]
            if weight > 1.0:
                # 为高误差特征降低阈值，以选择更多相关特征
                target_threshold = threshold / weight
        
        feature_selector.fit(X_train, y_train[target_col])
        importances = feature_selector.get_feature_importance()
        return [(X_train.columns[j], imp) for j, imp in enumerate(importances) if imp > target_threshold]
    
    # 并行执行特征重要性计算
    important_features_by_target = Parallel(n_jobs=n_jobs)(
        delayed(get_important_features_for_target)(col) for col in y_train.columns
    )
    
    # 合并所有目标的重要特征
    all_important_features = set()
    for target_features in important_features_by_target:
        all_important_features.update([f[0] for f in target_features])
    
    # 确保至少有10个特征
    if len(all_important_features) < 10:
        print(f"特征选择后特征数量过少 ({len(all_important_features)}), 使用默认重要特征")
        # 使用预设的重要特征列表
        default_features = [col for col in X_train.columns if 
                           'Error_' in col or 
                           'roll_mean' in col or 
                           'interaction' in col][:10]
        important_features = default_features
    else:
        important_features = list(all_important_features)
    
    # 特别确保包含与高误差特征相关的特征
    high_error_related_features = [col for col in X_train.columns if 
                                  any(err_prefix in col for err_prefix in ['Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_T_SONIC'])]
    
    # 将高误差相关特征加入选择的特征集
    important_features = list(set(important_features + high_error_related_features))
    
    print(f"选择了 {len(important_features)} 个重要特征，其中包含高误差特征相关特征")
    
    return X_train[important_features], X_val[important_features], X_test[important_features], important_features

# 加载原始数据集
_train_dataSet_raw = pd.read_csv("001-数据集\\真实值\\modified_数据集Time_Series661.dat")
_test_dataSet_raw = pd.read_csv("001-数据集\\不含真实值\\modified_数据集Time_Series662.dat")

print("原始训练集列名:", _train_dataSet_raw.columns.tolist())
print("原始测试集列名:", _test_dataSet_raw.columns.tolist())

# 查看原始数据 CL 列的缺失情况
data_for_missing_check = _train_dataSet_raw[CL]
missingDf = data_for_missing_check.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data_for_missing_check.shape[0]
print("原始数据 CL 列缺失值比例:")
print(missingDf)

# 对原始训练数据的 CL 列进行缺失值填充和异常值处理
_train_dataSet_cl_processed = _train_dataSet_raw.copy()
for col in CL:
    if col in _train_dataSet_cl_processed.columns:
        if pd.api.types.is_numeric_dtype(_train_dataSet_cl_processed[col]):
            pass
        else:
            _train_dataSet_cl_processed[col] = pd.to_numeric(_train_dataSet_cl_processed[col], errors='coerce')
            if _train_dataSet_cl_processed[col].isnull().all():
                print(f"Warning: Column {col} became all NaN after to_numeric conversion. Check data.")
            else:
                print(f"Warning: Column {col} was converted to numeric. Review data if unexpected.")
    else:
        print(f"Warning: Column {col} not found in _train_dataSet_cl_processed for initial numeric check.")

_train_dataSet_cl_processed[CL] = _train_dataSet_cl_processed[CL].fillna(_train_dataSet_cl_processed[CL].select_dtypes(include=np.number).median())
for column in CL:
    if column in _train_dataSet_cl_processed.columns and pd.api.types.is_numeric_dtype(_train_dataSet_cl_processed[column]):
        if not _train_dataSet_cl_processed[column].dropna().empty:
            z_scores = np.abs(stats.zscore(_train_dataSet_cl_processed[column].dropna()))
            outliers_original_indices = _train_dataSet_cl_processed[column].dropna()[z_scores > 2].index
            column_median = _train_dataSet_cl_processed[column].median()
            _train_dataSet_cl_processed.loc[outliers_original_indices, column] = column_median
        else:
            print(f"Warning: Column {column} is all NaN after dropna(), skipping outlier processing.")
    else:
        print(f"Warning: Column {column} for outlier processing not found or not numeric in training data.")

# 对原始测试数据的 CL 列进行缺失值填充 (使用训练集的中位数)
_test_dataSet_cl_processed = _test_dataSet_raw.copy()
for col in CL:
    if col in _test_dataSet_cl_processed.columns:
        if pd.api.types.is_numeric_dtype(_test_dataSet_cl_processed[col]):
            pass
        else:
            _test_dataSet_cl_processed[col] = pd.to_numeric(_test_dataSet_cl_processed[col], errors='coerce')

        if col in _train_dataSet_raw.columns and pd.api.types.is_numeric_dtype(_train_dataSet_raw[col]):
            train_median_for_col = _train_dataSet_raw[col].median()
            _test_dataSet_cl_processed[col] = _test_dataSet_cl_processed[col].fillna(train_median_for_col)
        else:
            numeric_test_col_median = _test_dataSet_cl_processed[col].select_dtypes(include=np.number).median()
            if not pd.isna(numeric_test_col_median):
                 _test_dataSet_cl_processed[col] = _test_dataSet_cl_processed[col].fillna(numeric_test_col_median)
            else:
                 _test_dataSet_cl_processed[col] = _test_dataSet_cl_processed[col].fillna(0)
            print(f"Warning: Column {col} used fallback median imputation for test set.")

# 应用增强的特征工程
train_dataSet = enhanced_preprocessing(_train_dataSet_cl_processed)
test_dataSet = enhanced_preprocessing(_test_dataSet_cl_processed)

print("处理后训练集列名:", train_dataSet.columns.tolist())
print("处理后测试集列名:", test_dataSet.columns.tolist())

candidate_X_columns = [col for col in train_dataSet.columns if
                       col.startswith('Error_') or
                       'rolling' in col or
                       'interaction' in col or
                       'cross_modality' in col or
                       col.endswith('_squared')]

updated_X_columns = [col for col in candidate_X_columns if col in test_dataSet.columns]

removed_cols = set(candidate_X_columns) - set(updated_X_columns)
if removed_cols:
    print(f"警告: 以下特征在训练集中生成，但在测试集中未能生成（或其依赖的原始列缺失），已从特征列表中移除: {list(removed_cols)}")
    print("请检查测试数据的原始列（例如 'CO2_sig_strgth', 'Error_T_SONIC'）是否存在且数据有效。")

updated_X_columns = sorted(list(set(updated_X_columns)))
if not updated_X_columns:
    raise ValueError("No feature columns selected for X. Check 'updated_X_columns' logic.")
print(f"Selected {len(updated_X_columns)} feature columns for X: {updated_X_columns}")

train_dataSet_sampled = train_dataSet.sample(frac=0.4, random_state=217)

X_train_full = train_dataSet_sampled[updated_X_columns]
y_train_full = train_dataSet_sampled[columns]
X_test = test_dataSet[updated_X_columns]

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=217)

X_train_selected, X_val_selected, X_test_selected, important_features = select_important_features(
    X_train, y_train, X_val, X_test
)

robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(X_train_selected)
X_val_scaled = robust_scaler.transform(X_val_selected)
X_test_scaled = robust_scaler.transform(X_test_selected)

base_models = [
    ('catboost_high_error', CatBoostRegressor(**{
        'iterations': 700,  # 为高误差特征增加迭代次数
        'learning_rate': 0.04,
        'depth': 8,  # 增加深度以捕捉复杂模式
        'l2_leaf_reg': 3,  # 减少正则化以更好拟合高误差特征
        'random_seed': 217,
        'loss_function': 'RMSE',
        'early_stopping_rounds': 70,
        'verbose': 0,
        'task_type': 'CPU',
        'thread_count': max(1, n_jobs // 3),
        'allow_writing_files': False
    })),
    ('lightgbm', LGBMRegressor(
        num_leaves=50,
        learning_rate=0.04,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.15,
        random_state=217,
        verbose=-1,
        n_jobs=max(1, n_jobs // 3)
    )),
    ('xgboost', XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=217,
        verbosity=0,
        n_jobs=max(1, n_jobs // 3)
    ))
]

stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=CatBoostRegressor(
        iterations=150,  # 减少迭代次数
        learning_rate=0.04,
        depth=5,
        verbose=0,
        random_seed=217,
        task_type='CPU',
        thread_count=n_jobs // 2,  # 使用一半的线程
        allow_writing_files=False
    ),
    n_jobs=n_jobs,
    cv=5  # 减少交叉验证折数
)

def train_target_specific_models(X_train_data, y_train_data):
    """针对不同误差水平的目标训练专用模型"""
    target_models = {}
    
    # 定义每个目标的专用参数，根据误差水平调整
    target_params = {
        'CO2_density_fast_tmpr': {  # 最高误差
            'model': 'catboost',
            'params': {
                'iterations': 900,
                'learning_rate': 0.03,
                'depth': 9,
                'l2_leaf_reg': 2,
                'random_seed': 217,
                'verbose': 0,
                'task_type': 'CPU',
                'thread_count': n_jobs // 3
            }
        },
        'CO2_density': {  # 高误差
            'model': 'catboost',
            'params': {
                'iterations': 800,
                'learning_rate': 0.03,
                'depth': 8,
                'l2_leaf_reg': 3,
                'random_seed': 217,
                'verbose': 0,
                'task_type': 'CPU',
                'thread_count': n_jobs // 3
            }
        },
        'T_SONIC': {  # 中高误差
            'model': 'xgboost',
            'params': {
                'n_estimators': 700,
                'learning_rate': 0.03,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.05,
                'reg_lambda': 0.1,
                'random_state': 217,
                'verbosity': 0,
                'n_jobs': n_jobs // 3
            }
        },
        'H2O_density': {  # 中等误差
            'model': 'lightgbm',
            'params': {
                'num_leaves': 50,
                'learning_rate': 0.04,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.2,
                'random_state': 217,
                'verbose': -1,
                'n_jobs': n_jobs // 3
            }
        }
    }
    
    def train_model_for_target(target_name):
        print(f"训练 {target_name} 专用模型...")
        
        # 检查是否有特定目标的参数配置
        if target_name in target_params:
            config = target_params[target_name]
            if config['model'] == 'catboost':
                model = CatBoostRegressor(**config['params'])
            elif config['model'] == 'lightgbm':
                model = LGBMRegressor(**config['params'])
            elif config['model'] == 'xgboost':
                model = XGBRegressor(**config['params'])
        else:
            # 默认配置 - 低误差特征使用更轻量的模型
            model = LGBMRegressor(
                num_leaves=30,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.7,
                reg_alpha=0.2,
                random_state=217,
                n_jobs=n_jobs // 3
            )
        
        model.fit(X_train_data, y_train_data[target_name])
        return target_name, model
    
    # 对所有目标列并行训练专用模型（不再只训练高误差特征）
    results = Parallel(n_jobs=min(len(columns), n_jobs))(
        delayed(train_model_for_target)(target) 
        for target in columns if target in y_train_data.columns
    )
    
    # 收集结果
    for target_name, model in results:
        target_models[target_name] = model
    
    return target_models

stacked_model = MultiOutputRegressor(stacking_regressor, n_jobs=-1)

def train_pipeline(model, X_train_data, y_train_data, X_val_data, y_val_data):
    """增强的训练流程"""
    print("训练主集成模型...")
    model.fit(X_train_data, y_train_data)
    
    print("训练专用目标模型...")
    target_models = train_target_specific_models(X_train_data, y_train_data)
    
    y_val_pred_main = model.predict(X_val_data)
    main_mse = mean_squared_error(y_val_data, y_val_pred_main)
    print(f"主模型验证集MSE: {main_mse:.6f}")
    
    target_preds = {}
    target_mses = {}
    for target_name, target_model in target_models.items():
        target_pred = target_model.predict(X_val_data)
        target_mse = mean_squared_error(y_val_data[target_name], target_pred)
        target_preds[target_name] = target_pred
        target_mses[target_name] = target_mse
        print(f"{target_name} 专用模型MSE: {target_mse:.6f}")
    
    return model, target_models

print("执行增强训练流程...")
final_model, target_specific_models = train_pipeline(
    stacked_model, 
    X_train_scaled, 
    y_train, 
    X_val_scaled,
    y_val
)

def enhanced_predict(main_model, target_models, X_data):
    """根据误差水平优化的预测函数"""
    main_preds = main_model.predict(X_data)
    
    if target_models:
        enhanced_preds = main_preds.copy()
        
        # 根据误差水平为每个特征设置不同的专用模型权重
        target_weights = {
            'CO2_density_fast_tmpr': 0.6,  # 最高误差 - 专用模型权重最高
            'CO2_density': 0.55,          # 高误差
            'T_SONIC': 0.5,               # 中高误差
            'H2O_density': 0.4,           # 中等误差
            'H2O_sig_strgth': 0.3,        # 低误差
            'CO2_sig_strgth': 0.3         # 低误差
        }
        
        for i, target_name in enumerate(columns):
            if target_name in target_models:
                target_pred = target_models[target_name].predict(X_data)
                # 使用基于误差水平的权重
                weight = target_weights.get(target_name, 0.4)  # 默认权重0.4
                enhanced_preds[:, i] = (1 - weight) * main_preds[:, i] + weight * target_pred
        
        return enhanced_preds
    else:
        return main_preds

y_val_pred = enhanced_predict(final_model, target_specific_models, X_val_scaled)
r2 = r2_score(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
print(f"增强预测 - 验证集 R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

y_predict = enhanced_predict(final_model, target_specific_models, X_test_scaled)

results = []
for Predicted_Value in y_predict:
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    results.append([formatted_predicted_value])

result_df = pd.DataFrame(results, columns=['Predicted_Value'])
result_df.to_csv("result_Enhanced_CatBoost_Stacked.csv", index=False)

print("预测完成，结果已保存至 result_Enhanced_CatBoost_Stacked.csv")

print("\n=== 模型性能摘要 ===")
print(f"总体 R2 得分: {r2:.4f}")
print(f"总体 MSE: {mse:.4f}")
print(f"总体 RMSE: {np.sqrt(mse):.4f}")

# 修改评估输出 - 针对高误差特征的详细分析
print("\n=== 按误差水平分组的模型性能 ===")

# 高误差特征组
high_error_targets = ['CO2_density_fast_tmpr', 'CO2_density', 'T_SONIC']
print("\n高误差特征组:")
for i, target_name in enumerate(columns):
    if target_name in high_error_targets:
        target_r2 = r2_score(y_val[target_name], y_val_pred[:, i])
        target_mse = mean_squared_error(y_val[target_name], y_val_pred[:, i])
        target_mae = mean_absolute_error(y_val[target_name], y_val_pred[:, i])
        print(f"{target_name}: R2={target_r2:.4f}, MSE={target_mse:.4f}, MAE={target_mae:.4f}")

# 中低误差特征组
low_error_targets = ['H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
print("\n中低误差特征组:")
for i, target_name in enumerate(columns):
    if target_name in low_error_targets:
        target_r2 = r2_score(y_val[target_name], y_val_pred[:, i])
        target_mse = mean_squared_error(y_val[target_name], y_val_pred[:, i])
        target_mae = mean_absolute_error(y_val[target_name], y_val_pred[:, i])
        print(f"{target_name}: R2={target_r2:.4f}, MSE={target_mse:.4f}, MAE={target_mae:.4f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")

