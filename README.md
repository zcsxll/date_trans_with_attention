# date_trans_with_attention
a pytorch implementation of machine translation model that translates human readable dates (eg: "25th of June, 2009", "Thursday, June 25, 2009") into machine readable dates ("2009-06-25")
LSTM and attention mechanism are used

# usage:
- python3 train.py (train with gpu)
- python2 train.py no_gpu (train with cpu)
- Python3 test.py (test th model)

# attention result
<p align="center">
<img src="https://github.com/zcsxll/date_trans_with_attention/blob/master/attention.png" width="600">
</p>
