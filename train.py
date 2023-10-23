from torch.utils.data import DataLoader, random_split
from dataset import IEEGDataset

if __name__ == "__main__":
    data_path = "data/preprocessed"
    participants = ['sub-%02d' % i for i in range(1, 11)]

    dataset = IEEGDataset(feat_path=data_path, participants=participants, preprocess_again=False)
    num_train = int(0.7 * len(dataset))
    num_val = int(0.2 * len(dataset))
    num_test = len(dataset) - num_train - num_val

    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    batch = next(iter(val_loader))
    print(batch[0].shape, batch[1].shape)
    print(batch[0], batch[1])
