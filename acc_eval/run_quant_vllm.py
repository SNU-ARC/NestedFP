from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = '/disk2/models/DeepSeek-R1'



model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="half",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
SAVE_DIR = '/disk2/models/DeepSeek-R1-FP8-Dynamic-Half'

oneshot(model=model, recipe=recipe, output_dir=SAVE_DIR)

# Save the model.
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
