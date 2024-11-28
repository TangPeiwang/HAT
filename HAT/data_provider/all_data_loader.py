from data_provider.all_dataset import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, \
    Dataset_M4, SegLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch


def data_provider(args, flag):
    global max_length
    max_length = args.pt_len

    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        batch_size = args.batch_size  # bsz for train and valid
    drop_last = False

    SMD_dataset = SegLoader(root_path='./dataset/SMD', win_size=100, flag=flag)
    MSL_dataset = SegLoader(root_path='./dataset/MSL', win_size=100, flag=flag)
    SMAP_dataset = SegLoader(root_path='./dataset/SMAP', win_size=100, flag=flag)
    SWAT_dataset = SegLoader(root_path='./dataset/SWAT', win_size=100, flag=flag)
    PSM_dataset = SegLoader(root_path='./dataset/PSM', win_size=100, flag=flag)

    seq_len = args.seq_len
    ETTh1_dataset = Dataset_ETT_hour(root_path='./dataset/ETT-small', data_path='ETTh1.csv', seq_len=seq_len)
    ETTh2_dataset = Dataset_ETT_hour(root_path='./dataset/ETT-small', data_path='ETTh2.csv', seq_len=seq_len)
    ETTm1_dataset = Dataset_ETT_minute(root_path='./dataset/ETT-small', data_path='ETTm1.csv', seq_len=seq_len)
    ETTm2_dataset = Dataset_ETT_minute(root_path='./dataset/ETT-small', data_path='ETTm2.csv', seq_len=seq_len)
    Electricity_dataset = Dataset_Custom(root_path='./dataset/electricity', data_path='electricity.csv',
                                         seq_len=seq_len)
    Exchange_dataset = Dataset_Custom(root_path='./dataset/exchange_rate', data_path='exchange_rate.csv',
                                      seq_len=seq_len)
    Illness_dataset = Dataset_Custom(root_path='./dataset/illness', data_path='national_illness.csv', seq_len=104)
    Traffic_dataset = Dataset_Custom(root_path='./dataset/traffic', data_path='traffic.csv', seq_len=seq_len)
    Weather_dataset = Dataset_Custom(root_path='./dataset/weather', data_path='weather.csv', seq_len=seq_len)

    seq_len = 96
    ETTh1_dataset1 = Dataset_ETT_hour(root_path='./dataset/ETT-small', data_path='ETTh1.csv', seq_len=seq_len)
    ETTh2_dataset1 = Dataset_ETT_hour(root_path='./dataset/ETT-small', data_path='ETTh2.csv', seq_len=seq_len)
    ETTm1_dataset1 = Dataset_ETT_minute(root_path='./dataset/ETT-small', data_path='ETTm1.csv', seq_len=seq_len)
    ETTm2_dataset1 = Dataset_ETT_minute(root_path='./dataset/ETT-small', data_path='ETTm2.csv', seq_len=seq_len)
    Electricity_dataset1 = Dataset_Custom(root_path='./dataset/electricity', data_path='electricity.csv',
                                          seq_len=seq_len)
    Weather_dataset1 = Dataset_Custom(root_path='./dataset/weather', data_path='weather.csv', seq_len=seq_len)

    # M4_Yearly_dataset = Dataset_M4(seasonal_patterns='Yearly')
    # M4_Monthly_dataset = Dataset_M4(seasonal_patterns='Monthly')
    # M4_Quarterly_dataset = Dataset_M4(seasonal_patterns='Quarterly')
    # M4_Weekly_dataset = Dataset_M4(seasonal_patterns='Weekly')
    # M4_Daily_dataset = Dataset_M4(seasonal_patterns='Daily')
    # M4_Hourly_dataset = Dataset_M4(seasonal_patterns='Hourly')

    all_dataset = ConcatDataset([
        SMD_dataset, MSL_dataset, SMAP_dataset, SWAT_dataset, PSM_dataset,
        ETTh1_dataset, ETTh2_dataset, ETTm1_dataset, ETTm2_dataset,
        # M4_Hourly_dataset, M4_Yearly_dataset, M4_Monthly_dataset, M4_Quarterly_dataset, M4_Weekly_dataset, M4_Daily_dataset,
        Electricity_dataset, Exchange_dataset, Illness_dataset, Traffic_dataset, Weather_dataset,
        ETTh1_dataset1, ETTh2_dataset1, ETTm1_dataset1, ETTm2_dataset1, Electricity_dataset1, Weather_dataset1
    ])
    data_loader = DataLoader(
        all_dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        collate_fn=GGBond_collate_fn,
        drop_last=drop_last)
    return data_loader


def GGBond_collate_fn(data):
    # max_length = 336
    sequences_padded = []
    sequences_mask = []
    for seq in data:
        sequences_padded.append(torch.nn.functional.pad(torch.tensor(seq), pad=(0, max_length - len(seq))))
        mask_zeros = torch.zeros(max_length - len(seq))
        mask_ones = torch.ones(len(seq))
        sequences_mask.append(torch.cat((mask_ones, mask_zeros)))

    return torch.stack(sequences_padded), torch.stack(sequences_mask)
