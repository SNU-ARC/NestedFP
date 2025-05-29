from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = '/disk/models/Llama-3.1-8B'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="half",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model.
SAVE_DIR = '/disk/models/Llama-3.1-8B-FP8-Dynamic-Half'
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
