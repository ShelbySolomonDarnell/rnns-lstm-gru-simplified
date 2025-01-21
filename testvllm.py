import torch
from common import *
from vllm import LLM, SamplingParams

prompts = [
  "Where art thou", 
  "Romeo and Juliet",
]

f_name = inspect.stack()[0][3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tellem.info("[{0}] Device {1}".format(f_name, device))
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40)

#llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct-q4_0", gpu_memory_utilization=0.95)
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.5, device=device)
responses = llm.generate(prompts, sampling_params)

for response in responses:
    print(response.prompt)
    print(response.outputs[0].text)

