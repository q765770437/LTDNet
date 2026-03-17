import argparse
import os

import torch
from experiments.exp_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

if __name__ == '__main__':
    # seed = 2025
    seed = 2024
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(description='LTDNet')

    # 基础配置
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, mask, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')  # 新参数中存在，保留
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')  # 新参数中存在，保留
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                        help='model name, options: [DFGCN, iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')  # 覆盖原有model选项

    # 数据加载
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')  # 覆盖原有默认值
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')  # 覆盖原有路径
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')  # 覆盖原有文件名
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # 预测任务配置
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length (no longer needed in inverted Transformers)')  # 补充说明
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # 原有预测任务中独有的参数（新参数未覆盖，保留）
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock/ScaleGraphBlock')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--num_nodes', type=int, default=7, help='to create Graph')
    parser.add_argument('--subgraph_size', type=int, default=3, help='neighbors number')
    parser.add_argument('--tanhalpha', type=float, default=3, help='')

    # GCN配置（新参数未涉及，保留）
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    # DLinear配置（新参数未涉及，保留）
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # 模型定义
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size (applicable on arbitrary number of variates in inverted Transformers)')  # 补充说明
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')  # 覆盖原有默认值
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')  # 覆盖原有默认值
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')  # 覆盖原有默认值
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')  # 覆盖原有默认值
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')  # 覆盖原有默认值
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')  # 覆盖原有默认值
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')  # 覆盖原有默认值
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='sigmoid', help='activation')  # 覆盖原有默认值
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')  # 修正拼写错误（ecoder→encoder）
    # 原有Formers中独有的参数（新参数未覆盖，保留）
    parser.add_argument('--embed_type', type=int, default=0, help='0: default '
                                                                  '1: value embedding + temporal embedding + positional embedding '
                                                                  '2: value embedding + temporal embedding '
                                                                  '3: value embedding + positional embedding '
                                                                  '4: value embedding')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # 优化器配置
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')  # 覆盖原有默认值
    parser.add_argument('--itr', type=int, default=1, help='experiments times')  # 覆盖原有默认值
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')  # 新参数独有，添加

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')  # 修正拼写错误（multile→multiple）
    # 原有GPU配置中独有的参数（新参数未覆盖，保留）
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Transformer特有配置（新参数独有，全部添加）
    parser.add_argument('--use_norm', type=bool, default=True, help='use norm and denorm')
    parser.add_argument('--exp_name', type=str, required=False, default='None',
                        help='experiment name, options:[partial_train, zero_shot]')  # 修正拼写错误（experiemnt→experiment）
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
    parser.add_argument('--k', type=int, default=2, help='similarity threshold')
    parser.add_argument('--patch_len', type=int, default=8, help='patch len')
    parser.add_argument('--d_core', type=int, default=128, help='patch len')
    parser.add_argument('--kernel_size', type=int, default=3, help='小波变换的核大小')
    parser.add_argument('--cross_kernel_size', type=int, default=3, help='跨变量关系卷积核大小')
    parser.add_argument('--k_factor', type=float, default=5, help='卷积核宽度乘的系数')
    parser.add_argument('--mark', type=int, default=1, help='是否有xmark')
    parser.add_argument('--vi', type=bool, default=False, help='是否输出v')
    parser.add_argument('--use_loss', type=bool, default=True, help='是否使用loss')
    parser.add_argument('--noise_scale', type=float, default=0, help='是否加噪声')  # 新参数独有，添加
    parser.add_argument('--mem_size', type=int, default=3, help='记忆单元长度')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)


    if args.exp_name == 'partial_train':
        Exp = Exp_Long_Term_Forecast_Partial
    else:
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_patch{}_k{}_cs{}_des{}_ii{}_kers_{}_cores_{}'.format(
                args.model_id,          # 模型ID
                args.model,             # 模型名称
                args.data,              # 数据集类型
                args.features,          # 特征类型（M/S/MS）
                args.seq_len,           # 输入序列长度
                args.label_len,         # 标签序列长度
                args.pred_len,          # 预测序列长度
                args.d_model,           # 模型维度
                args.n_heads,           # 注意力头数
                args.e_layers,          # 编码器层数
                args.d_layers,          # 解码器层数
                args.d_ff,              # 前馈网络维度
                args.factor,            # 注意力因子
                args.embed,             # 嵌入类型
                args.distil,            # 是否蒸馏
                args.patch_len,         # 新增：patch长度
                args.k,                 # 新增：相似度阈值
                args.class_strategy,    # 分类策略（projection/average等）
                args.des,               # 实验描述
                ii,
                args.kernel_size,
                args.d_core# 迭代次数
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_patch{}_k{}_cs{}_des{}_ii{}_kers_{}_cores_{}'.format(
            args.model_id,          # 模型ID
            args.model,             # 模型名称
            args.data,              # 数据集类型
            args.features,          # 特征类型（M/S/MS）
            args.seq_len,           # 输入序列长度
            args.label_len,         # 标签序列长度
            args.pred_len,          # 预测序列长度
            args.d_model,           # 模型维度
            args.n_heads,           # 注意力头数
            args.e_layers,          # 编码器层数
            args.d_layers,          # 解码器层数
            args.d_ff,              # 前馈网络维度
            args.factor,            # 注意力因子
            args.embed,             # 嵌入类型
            args.distil,            # 是否蒸馏
            args.patch_len,         # 新增：patch长度
            args.k,                 # 新增：相似度阈值
            args.class_strategy,    # 分类策略（projection/average等）
            args.des,               # 实验描述
            ii,
            args.kernel_size,
            args.d_core  # 迭代次数
            # 迭代次数
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()