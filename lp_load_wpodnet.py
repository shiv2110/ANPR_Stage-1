from os.path import splitext
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model



def load_save_model(path):
    try:
        path_root = splitext(path)[0]
        with open(path, 'r') as json_file:
            model_json_file = json_file.read()
        model = model_from_json(model_json_file, custom_objects = {})
        model.load_weights(path_root + ".h5")
        model.save("model/wpod-net-combined.h5")
        print("Model has been saved successfully")
        return model
    except Exception as e:
        print(e)


arch_path = "model/wpod-net.json"
weights_and_arch_wpodNet = load_save_model(arch_path)
model_lpr = load_model("model/wpod-net-combined.h5")

def get_model_wpodnet():
    return model_lpr