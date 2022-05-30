from get_data import GetData

if __name__ == '__main__':
    print('Getting data...')
    
    data = GetData.get_stocks_data()

    df = data[:int(len(data)*.7)]
    df_test = data[int(len(data)*.7):]

    print('Train data shape =', df.shape)
    print('Test data shape =', df_test.shape)

    df.to_csv('../../../../data/df_train.csv', index=False, encoding='utf-8')
    df_test.to_csv('../../../../data/df_test.csv', index=False, encoding='utf-8')

    print('Process finished')
