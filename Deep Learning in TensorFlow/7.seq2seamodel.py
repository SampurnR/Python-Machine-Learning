import seq2seq
from seq2seq.models import SimpleSeq2Seq
model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=20, depth=(4, 5))
model.compile(loss='mse', optimizer='rmsprop')