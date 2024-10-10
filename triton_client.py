import numpy as np
import tritonclient.http as http_client
from transformers import AutoTokenizer

tokenzier = AutoTokenizer.from_pretrained("bert-base-uncased")

triton_client = http_client.InferenceServerClient(
    url="localhost:8000",
)

input_names = ["input__0", "input__1"]
output_names = ["output__0"]
data_type = np.int32

def run_inference(sentence):
    inputs = tokenzier(sentence, return_tensors='pt', max_length=256, padding="max_length")
    required_shape = (1, 256)
    input_ids = inputs['input_ids'].numpy().reshape(required_shape).astype(data_type)
    attention_mask = inputs['attention_mask'].numpy().reshape(required_shape).astype(data_type)
    
    input_0 = http_client.InferInput(input_names[0], required_shape, "INT32")
    input_0.set_data_from_numpy(input_ids)

    input_1 = http_client.InferInput(input_names[1], required_shape, "INT32")
    input_1.set_data_from_numpy(attention_mask)

    output = http_client.InferRequestedOutput(output_names[0])
    
    results = triton_client.infer(model_name='bert', inputs=(input_0, input_1), outputs=(output, ))

    output_data = results.as_numpy("output__0")
    return output_data

sentence = "test sentence"
print(run_inference(sentence))

