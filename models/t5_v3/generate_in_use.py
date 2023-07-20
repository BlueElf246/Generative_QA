from utils import *
model_path = "model_3_ans"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained("t5-base")
question = 'What is color means?'
context = "Color is a perceptual property of light that is typically associated with the visual " \
          "experience of humans and many other animals. It is a result of the interaction between light," \
          " objects, and the human visual system. Light is made up of electromagnetic waves that travel in " \
          "different wavelengths. The different wavelengths of light correspond to different colors. " \
          "When light strikes an object, some wavelengths are absorbed by the object, " \
          "while others are reflected or transmitted. The wavelengths that are reflected or " \
          "transmitted are detected by the human eye, which contains specialized cells called cones " \
          "that are sensitive to different ranges of wavelengths. The human eye has three types of cones: " \
          "those that are most sensitive to short wavelengths (which we perceive as blue), "

generate_in_use(question, context, model, tokenizer)