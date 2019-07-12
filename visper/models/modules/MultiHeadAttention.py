import torch
from torch import nn
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    def __init__(self, attention_size, in_features, residual=True):
        """
            Args:
                attention_size - length of attention vector
        """
        super(MultiHeadAttention, self).__init__()
        self.residual = residual


        self.relu = nn.ReLU(inplace=True    )
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.query_dense = nn.Linear(in_features, attention_size)
        self.key_dense = nn.Linear(in_features, attention_size)
        self.value_dense = nn.Linear(in_features, attention_size)
    
    def forward(self, x):
        """
            Args:
                x - encoder output of shape batch_size X time X features
        """

        Q_ = self.query_dense(x)
        #Q_ = self.relu(Q_)
        #self.relu(Q_)

        K_ = self.key_dense(x)
        #K_ = self.relu(K_)
        #self.relu(K_)

        V_ = self.value_dense(x)
        #V_ = self.relu(V_)
        #self.relu(V_)

        K_transpose = K_.transpose(2,1)
        mm = torch.bmm(Q_, K_transpose)

        alignment = self.softmax(mm)
        #alignment = self.sigmoid(mm)

        output = torch.bmm(alignment, V_)

        if self.residual:
            output += x

        return output, alignment


if __name__ == "__main__":
    encoder_out = torch.zeros(64, 29, 512, dtype=torch.float32).random_(to=3)

    attention = MultiHeadAttention(attention_size=512, in_features=512, residual=True)
    out, align = attention(encoder_out)

    align_map = align[0].cpu().detach().numpy()

    plt.imshow(align_map)
    plt.show()



    print()