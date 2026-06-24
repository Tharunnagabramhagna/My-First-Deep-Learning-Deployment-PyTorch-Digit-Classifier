from flask import Flask, request, jsonify

from app.torch_utils import transform_Image, get_prediction

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    # E.g:- FILENAME.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    # after resplit we get => .png so start from index-1


@app.route('/predict', methods=['POST'])
def predict():
    # Safe Checks
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_Image(img_bytes)
            prediction = get_prediction(tensor)
            # str(digit) => E.g :- '0','1','2',etc...
            data = {'predicition': prediction.item(
            ), 'class_name': str(prediction.item())}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

    # 1) load image
    # 2) image -> tensor
    # 3) prediction
    # 4) return json
    return jsonify({'result': 1})
