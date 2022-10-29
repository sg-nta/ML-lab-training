import DataReader as dr

def load_datasets():
    with open("../datasets/20news-bydate/words_idfs.txt") as f:
        vocab_size = len(f.read().splitlines())
        
    train_data_reader = dr.DataReader(
        data_path='../datasets/20news-bydate/20news-train-tf-idf.txt',
        batch_size=50,
        vocab_size = vocab_size
    )

    test_data_reader = dr.DataReader(
        data_path='../datasets/20news-bydate/20news-test-tf-idf.txt',
        batch_size=50,
        vocab_size = vocab_size
    )

    return train_data_reader, test_data_reader

def save_parameters(name, value, epoch):
    filename = name.replace(':', '-colon-') + f'-epoch-{epoch}.txt'
    if len(value.shape) == 1:
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number)
                                            for number in value[row]])
                                            for row in range(value.shape[0])])
        
    with open(f'../datasets/20news-bydate/saved-params/{filename}', 'w') as f:
        f.write(string_form)

def restore_parameters(name, epoch):
    filename = name.replace(':', '-colon-') + f'-epoch-{epoch}.txt'
    with open(f'../datasets/20news-bydate/saved-params/{filename}') as f:
        lines = f.read().splitlines()
    if len(lines) == 1:
        value = [[float(number) for number in lines[0].split(',')]]
    else:
        value = [[float(number) for number in lines[row].split(',')]
                for row in range(len(lines))]
    return value
