import torch

class Encoder(torch.nn.Module):
    def __init__(self, in_features=37, hidden_size=64):
        super(Encoder, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=hidden_size)
        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, in_features=128, hidden_size=64):
        super(Decoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=11)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, h, c):
        self.lstm.flatten_parameters()
        x, (h, c) = self.lstm(x, (h, c))
        x = self.linear(x.squeeze())
        x = self.softmax(x)
        return x, (h, c)

class Model(torch.nn.Module):
    def __init__(self, output_len):
        super(Model, self).__init__()
        self.output_len = output_len
        self.encoder = Encoder(in_features=37, hidden_size=64)
        self.decoder = Decoder(in_features=128, hidden_size=64)
        
        #attention net
        self.linear1 = torch.nn.Linear(in_features=128+64, out_features=32)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(in_features=32, out_features=1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        每次attention都使用decoder输出的完整feature以及decoder的h（初始为0）拼起来作为attention网络的输入
        '''
        feats = self.encoder(x) #(N, 30, 128)
        decoder_h = torch.zeros((1, feats.shape[0], 64)) #(N, 64)
        decoder_c = torch.zeros((1, feats.shape[0], 64)) #(N, 64)
        # print(feats.shape, decoder_h.shape)
        outputs = []
        for _ in range(self.output_len):
            feats_2 = decoder_h.transpose(0, 1) #(1, N, 64) --> (N, 1, 64)
            feats_2 = feats_2.repeat(1, feats.shape[1], 1) #(N, 1, 64) --> (N, 30, 64), cuz feats.shape[1] is 30
            # print(i, feats.shape, decoder_h.shape, feats_2.shape)
            feats_in = torch.cat((feats,feats_2), dim=-1) # (N, 30, 128) and (N, 30, 64) --> (N, 30, 192)
            out = self.tanh(self.linear1(feats_in)) #(N, 30, 192) --> (N, 30, 32)
            scores = self.softmax(self.linear2(out)) #(N, 30, 32) --> (N, 30, 1)
            
            # feats_for_decoder = []
            # for feat, score in zip(feats, scores):
            #     # print(feat.shape, score.shape)
            #     feats_for_decoder.append(torch.matmul(score, feat).view(1, -1))
            # feats_for_decoder = torch.cat(feats_for_decoder).unsqueeze(1)
            # print(feats_for_decoder.shape)

            feats_for_decoder = torch.mul(feats, scores).sum(axis=1).unsqueeze(1)

            # check the dot result
            # for b in range(4):
            #     for i in range(128):
            #         tmp1 = feats[b, :, i]
            #         tmp2 = scores[b].squeeze()
            #         print(tmp1.shape, tmp2.shape)
            #         tmp3 = tmp1.dot(tmp2)
            #         aa = tmp3.detach().numpy()
            #         bb = feats_for_decoder[b][0][i].detach().numpy()
            #         # print(aa, bb)
            #         assert abs(aa - bb) < 0.000001, f'{aa} {bb}'

            decoder_out, (decoder_h, decoder_c) = self.decoder(feats_for_decoder, decoder_h, decoder_c)
            outputs.append(decoder_out.unsqueeze(1)) #list of (N, 1, 11)
            # print(decoder_out.shape)
        outputs = torch.cat(outputs, dim=1) #(N, 10, 11) 10是输出序列长度
        # print(outputs.shape)
        return outputs

if __name__ == '__main__':
    import dataset

    dataset = dataset.Dataset(transform=dataset.transform, n_datas=100)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=None)

    model = Model(output_len=10)
    for step, (batch_x, batch_y) in enumerate(dataloader):
        pred = model(batch_x)
        print(pred.shape, batch_y.shape)
        break