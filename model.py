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
    def __init__(self, in_features=128, hidden_size=128):
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
    def __init__(self, output_len, use_gpu=True):
        super(Model, self).__init__()
        self.output_len = output_len
        self.use_gpu = use_gpu
        self.encoder = Encoder(in_features=37, hidden_size=64)
        self.decoder = Decoder(in_features=128, hidden_size=128)
        
        #attention net
        self.linear1 = torch.nn.Linear(in_features=128+128, out_features=10)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(in_features=10, out_features=1)
        self.softmax = torch.nn.Softmax(dim=1) #this is important

        self.scores_for_paint = []

    def forward(self, x):
        '''
        每次attention都使用decoder输出的完整feature以及decoder的h（初始为0）拼起来作为attention网络的输入
        '''
        feats = self.encoder(x) #(N, 30, 128)
        if self.use_gpu:
            decoder_h = torch.zeros((1, feats.shape[0], 128)).cuda() #(1, N, 128)
            decoder_c = torch.zeros((1, feats.shape[0], 128)).cuda() #(1, N, 128)
        else:
            decoder_h = torch.zeros((1, feats.shape[0], 128)) #(1, N, 128)
            decoder_c = torch.zeros((1, feats.shape[0], 128)) #(1, N, 128)
        # print(feats.shape, decoder_c.shape)
        outputs = []
        self.scores_for_paint = []
        for _ in range(self.output_len):
            feats_2 = decoder_c.transpose(0, 1) #(1, N, 128) --> (N, 1, 128)
            feats_2 = feats_2.repeat(1, feats.shape[1], 1) #(N, 1, 128) --> (N, 30, 128), cuz feats.shape[1] is 30
            feats_in = torch.cat((feats, feats_2), dim=-1) # (N, 30, 128) and (N, 30, 128) --> (N, 30, 256)
            # print(feats.shape, feats_2.shape, feats_in.shape)
            out = self.tanh(self.linear1(feats_in)) #(N, 30, 256) --> (N, 30, 10)
            scores = self.softmax(self.linear2(out)) #(N, 30, 10) --> (N, 30, 1)
            # print(scores[0, :, 0])

            if not self.training:
                self.scores_for_paint.append(scores.squeeze().detach().cpu().numpy())
                # print(scores)
            
            # feats_for_decoder = []
            # for feat, score in zip(feats, scores):
            #     # print(feat.shape, score.shape)
            #     feats_for_decoder.append(torch.matmul(score, feat).view(1, -1))
            # feats_for_decoder = torch.cat(feats_for_decoder).unsqueeze(1)
            # print(feats_for_decoder.shape)

            feats_for_decoder = torch.mul(feats, scores).sum(axis=1).unsqueeze(1) #(N, 30, 128) mul (N, 30, 1) --> (N, 30, 128) --> (N, 128) --> (N, 1, 128)
            # print(feats.shape, scores.shape, feats_for_decoder.shape)

            decoder_out, (decoder_h, decoder_c) = self.decoder(feats_for_decoder, decoder_h, decoder_c) #run lstm only one step, cuz feats_for_decoder.shape is (N, 1, 128)
            outputs.append(decoder_out.unsqueeze(1)) #list of (N, 1, 11)
            # print(decoder_out.shape)
        outputs = torch.cat(outputs, dim=1) #(N, 10, 11) 10是输出序列长度
        # print(outputs.shape)
        return outputs

    def total_parameters(self):
        return sum([p.numel() for p in self.parameters()])

if __name__ == '__main__':
    import dataset

    dataset = dataset.Dataset(transform=dataset.transform, n_datas=100)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=None)

    model = Model(output_len=10, use_gpu=False)
    print('model size is %.3f KB' % (model.total_parameters() * 4 / 1024))
    for step, (batch_x, batch_y, _) in enumerate(dataloader):
        pred = model(batch_x)
        print(pred.shape, batch_y.shape)
        break
