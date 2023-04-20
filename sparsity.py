from transformers import AutoModel, AutoTokenizer
import numpy as np

# Load the model and tokenizer from Hugging Faces
model_name = "20dc01/pruned-minilm"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Calculate the sparsity
total_params = 0
zero_params = 0

for name, param in model.named_parameters():
    if "bias" not in name:
        total_params += np.prod(param.shape)
        zero_params += np.count_nonzero(param.cpu().detach().numpy() == 0)

sparsity = zero_params / total_params
print(f"Final sparsity of {model_name}: {sparsity:.4f}")
