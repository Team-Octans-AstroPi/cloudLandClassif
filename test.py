from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file
import os

script_dir = Path(__file__).parent.resolve()

model_file = script_dir/'models/astropi-land.tflite' # name of model
data_dir = script_dir/'Land Tests'
label_file = script_dir/'Land Dataset'/'land-labels.txt' # Name of your label file

interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

size = common.input_size(interpreter)

correct_classifications = 0
all_classifications = 0

for cloud_type in os.listdir(data_dir):
    if cloud_type[0] != ".":
        cloud_path = os.path.join(data_dir, cloud_type)
        for image_name in os.listdir(os.path.join(data_dir, cloud_type)):
            if image_name[0] != ".":
                image_file =  os.path.join(cloud_path, image_name)
                image = Image.open(image_file).convert('L').resize(size, Image.ANTIALIAS).convert('RGB')

                common.set_input(interpreter, image)
                interpreter.invoke()
                classes = classify.get_classes(interpreter, top_k=1)

                labels = read_label_file(label_file)
                if labels.get(classes[0].id, classes[0].id) == str(cloud_type):
                    correct_classifications += 1
                #else:
                    #print(cloud_type, labels.get(classes[0].id, classes[0].id))
                all_classifications += 1

data_dir = script_dir/'Land Dataset'

for cloud_type in os.listdir(data_dir):
    if cloud_type[0] != "." and cloud_type != "land-labels.txt":
        cloud_path = os.path.join(data_dir, cloud_type)
          for image_name in os.listdir(os.path.join(data_dir, cloud_type)):
              if image_name[0] != ".":
                  image_file =  os.path.join(cloud_path, image_name)
                  image = Image.open(image_file).convert('L').resize(size, Image.ANTIALIAS).convert('RGB')

                  common.set_input(interpreter, image)
                  interpreter.invoke()
                  classes = classify.get_classes(interpreter, top_k=1)

                  labels = read_label_file(label_file)
                  if labels.get(classes[0].id, classes[0].id) == str(cloud_type):
                      correct_classifications += 1
                  #else:
                      #print(cloud_type, labels.get(classes[0].id, classes[0].id))
                  all_classifications += 1

print(correct_classifications / all_classifications, correct_classifications, all_classifications)
