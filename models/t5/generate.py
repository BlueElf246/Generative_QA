from utils import *

model, tokenizer = load_model(path= '/Users/datle/Desktop/code_structure/models/t5/lightning_logs/version_0/checkpoints/epoch=0-step=4119.ckpt')


question = 'what is color'
context = "Color is a perceptual property of light that is typically associated with the visual " \
          "experience of humans and many other animals. It is a result of the interaction between light," \
          " objects, and the human visual system. Light is made up of electromagnetic waves that travel in " \
          "different wavelengths. The different wavelengths of light correspond to different colors. " \
          "When light strikes an object, some wavelengths are absorbed by the object, " \
          "while others are reflected or transmitted. The wavelengths that are reflected or " \
          "transmitted are detected by the human eye, which contains specialized cells called cones " \
          "that are sensitive to different ranges of wavelengths. The human eye has three types of cones: " \
          "those that are most sensitive to short wavelengths (which we perceive as blue), " \
          "those sensitive to medium wavelengths (perceived as green), and those sensitive to long wavelengths (perceived as red). The combination and intensity of signals from these cones allow us to perceive a wide range of colors. Color perception can also be influenced by factors such as lighting conditions, the surrounding environment, and individual differences in color vision. Additionally, color is subjective and can evoke different emotions and associations in different individuals and cultures. In summary, color is a visual sensation that arises from the interaction of light, objects, and the human visual system, allowing us to perceive and differentiate various hues and shades."

answer = generate(model, tokenizer, question, context)

print(answer)