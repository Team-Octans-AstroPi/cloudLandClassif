from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

script_dir = Path(__file__).parent.resolve()

model_file = script_dir/'models/astropi-clouds.tflite' # name of model
data_dir = script_dir/'data'
label_file = data_dir/'cloudclasses' # Name of your label file
image_file = data_dir/'cumulonimbus'/'photo_00731_51844705457_o.jpg' # Name of image for classification

interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

size = common.input_size(interpreter)
image = Image.open(image_file).convert('L').resize(size, Image.ANTIALIAS).convert('RGB')

common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

labels = read_label_file(label_file)
for c in classes:
    print(f'{labels.get(c.id, c.id)} {c.score:.5f}')