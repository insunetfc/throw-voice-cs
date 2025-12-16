import torch
import numpy as np

# your extracted codes from the DAC step
codes = np.load("fake.npy")        # shape should be (10, 689) in your case

# make it a 2D tensor on CPU
codes_tensor = torch.tensor(codes, dtype=torch.long)   # shape [10, 689]

# now wrap it in a *list of length 1*, just like cached_ref.pt
prompt_tokens = [codes_tensor]
prompt_texts = [""]   # length 1

torch.save(
    {
        "prompt_tokens": prompt_tokens,
        "prompt_texts": prompt_texts,
    },
    "fake_ref.pt",
)
print("saved fake_ref.pt:", codes_tensor.shape)
