import math
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    This class represents the encoder component of the ViT model.
    
    Args:
    num_hidden_layers: Number of hidden layers
    hidden_size: Hidden size
    intermediate_size: Intermediate size
    num_attention_heads: Number of attention heads. Defaults to 4.
    attention_probs_dropout_prob: Dropout probability for attention. Defaults to 0.0.
    hidden_dropout_prob: Dropout probability for hidden layers. Defaults to 0.0.
    qkv_bias: Whether to use bias for query, key and value. Defaults to True.

    Returns:
    X: Tensor containing the output of the encoder
    """
    def __init__(self, num_hidden_layers, hidden_size, intermediate_size, num_attention_heads=4, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0, qkv_bias=True):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, qkv_bias).to(device)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        return X



class EncoderBlock(nn.Module):
    """
    Represents an encoder block within the ViT model.

    Args:
    hidden_size: Hidden size.
    intermediate_size: Intermediate size.
    num_attention_heads: Number of attention heads. Defaults to 4.
    attention_probs_dropout_prob: Dropout probability for attention. Defaults to 0.0.
    hidden_dropout_prob: Dropout probability for hidden layers. Defaults to 0.0.
    qkv_bias: Whether to use bias for query, key and value. Defaults to True.

    Returns:
    X: Tensor containing the output of the encoder block
    """
    def __init__(self, hidden_size, intermediate_size, num_attention_heads=4, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0, qkv_bias=True):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.MHA = MultiHeadAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, qkv_bias).to(device)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_dropout_prob).to(device)
    
    def forward(self, x):
        x = x + self.MHA(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MLP(nn.Module):
    """
    Represents a multi-layer perceptron within the ViT model.

    Args:
    hidden_size: Hidden size.
    intermediate_size: Intermediate size.
    hidden_dropout_prob: Dropout probability for hidden layers. Defaults to 0.0.

    Returns:
    X: Tensor containing the output of the MLP.
    """
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, X):
        X = self.fc1(X)
        X = self.GELU(X)
        X = self.fc2(X)
        X = self.dropout(X)
        return X



class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism for the ViT model.

    Args:
    hidden_size: The dimension of the hidden layers.
    attention_head_size: The size of each attention head.
    dropout_prob: The dropout probability for attention. Defaults to 0.0.
    bias: Whether to use bias in the linear layers. Defaults to True.

    Returns:
    A tuple containing the attention output and attention probabilities.
    """
    def __init__(self, hidden_size, attention_head_size, dropout_prob=0.0, bias=True):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, X):
        query = self.query(X)
        key = self.key(X)
        value = self.value(X)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)
        return attention_output, attention_probs



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for the ViT model.

    Args:
    hidden_size: The dimension of the hidden layers.
    num_attention_heads: The number of attention heads. Defaults to 4.
    attention_probs_dropout_prob: The dropout probability for attention. Defaults to 0.0.
    hidden_dropout_prob: The dropout probability for hidden layers. Defaults to 0.0.
    qkv_bias: Whether to use bias for query, key, and value. Defaults to True.

    Returns:
    attention_output: The output of the multi-head attention mechanism.
    """
    def __init__(self, hidden_size, num_attention_heads=4, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0, qkv_bias=True):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size

        self.qkv_bias = qkv_bias
        self.heads = nn.ModuleList([
            ScaledDotProductAttention(hidden_size, self.attention_head_size, attention_probs_dropout_prob, qkv_bias)
            for _ in range(num_attention_heads)
        ])
        
        self.output_projection = nn.Linear(self.all_head_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, X):
        attention_outputs = [head(X) for head in self.heads]
        attention_output = torch.cat([output[0] for output in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        return attention_output



class PatchEmbeddings(nn.Module):
    """
    Converts image patches into embeddings for the ViT model.

    Args:
    image_size: The size of the input image.
    hidden_size: The dimension of the hidden layers.
    patch_size: The size of each patch. Defaults to 16.
    num_channels: The number of channels in the input image. Defaults to 3 (RGB).

    Returns:
    X: The tensor containing the output of the patch embeddings.
    """
    def __init__(self, image_size, hidden_size, patch_size=16, num_channels=3):
        super(PatchEmbeddings, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.num_patches = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self, X):
        X = self.projection(X)
        X = X.flatten(2)
        X = X.transpose(-1, -2)
        return X



class Embeddings(nn.Module):
    """
    Embeddings layer for the ViT model, which includes patch embeddings and position embeddings.

    Args:
    hidden_size: The dimension of the hidden layers.
    image_size: The size of the input image.
    patch_size: The size of each patch. Defaults to 16.
    num_channels: The number of channels in the input image. Defaults to 3 (RGB).
    hidden_dropout_prob: The dropout probability for hidden layers. Defaults to 0.0.

    Returns:
    X: The tensor containing the output of the embeddings.
    """
    def __init__(self, hidden_size, image_size, patch_size=16, num_channels=3, hidden_dropout_prob=0.0):
        super(Embeddings, self).__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, hidden_size, patch_size, num_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, X):
        X = self.patch_embeddings(X)
        batch_size, _, _ = X.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        X = X + self.position_embeddings
        X = self.dropout(X)
        return X



class ViT(nn.Module):
    """
    Vision Transformer model.

    Args:
    image_size: The size of the input image.
    hidden_size: The dimension of the hidden layers.
    num_hidden_layers: The number of hidden layers.
    intermediate_size: The dimension of the intermediate layers.
    num_classes: The number of classes in the classification task.
    num_attention_heads: The number of attention heads. Defaults to 4.
    hidden_dropout_prob: The dropout probability for hidden layers. Defaults to 0.0.
    attention_probs_dropout_prob: The dropout probability for attention. Defaults to 0.0.
    num_channels: The number of channels in the input image. Defaults to 3 (RGB).
    patch_size: The size of each patch. Defaults to 16.
    qkv_bias: Whether to use bias for query, key, and value. Defaults to True.

    Returns:
    logits: The tensor containing the logits for classification.
    """
    def __init__(self, image_size, hidden_size, num_hidden_layers, intermediate_size, num_classes, num_attention_heads=4, 
                 hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, num_channels=3, patch_size=16, qkv_bias=True):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(hidden_size, image_size, patch_size, num_channels, hidden_dropout_prob)
        self.encoder = Encoder(num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, 
                               attention_probs_dropout_prob, hidden_dropout_prob, qkv_bias)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)

    def forward(self, X):
        embedding_output = self.embeddings(X)
        encoder_output = self.encoder(embedding_output)
        logits = self.classifier(encoder_output[:, 0, :])
        return logits
    
    def _init_weights(self, module, initializer_range=0.02):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(module.position_embeddings.data.to(torch.float32), 
                                                                    mean=0.0, 
                                                                    std=initializer_range).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data.to(torch.float32), 
                                                          mean=0.0, 
                                                          std=initializer_range).to(module.cls_token.dtype)



if __name__ == "__main__":
    model = ViT(224, 48, 4, 4 * 48, 3)
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device {param.device}")