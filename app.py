from flask import Flask, jsonify, request
import jsonschema
import json
from jsonschema import validate
from init_model import process_prompt
from config import Config

import sys
import logging



# JSON Schema для запроса
request_schema = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "n_images": {"type": "integer"},
        "gfpgan": {"type": "boolean"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "sampler": {"type": "string"},
        "nsfw": {"type": "boolean"},
        "num_inference_steps": {"type": "integer"},
        "guidance_scale": {"type": "number"},
        "device": {"type": "string"},
        "image": {"type": "string"},
        "scale": {"type": "integer"},
        "lora": {"type": "string"}
    },
    "required": ["prompt", "negative_prompt"]
}


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    @app.route("/")
    def index():
        return "API for predict service"

    @app.route('/predict', methods=['POST'])
    def generate_image():
        try:
            request_data = request.get_json()
            validate(request_data, request_schema)  # валидация запроса

            # обработка запроса
            result = process_prompt(request_data)
            if result:
                return json.dumps({"result": result}), 200
            else:
                return "Резльтат не сфомирован"

        except jsonschema.exceptions.ValidationError as e:
            # обработка ошибок валидации
            return jsonify({"Некорректный запрос": str(e)}), 400

        except Exception as e:
            # обработка других ошибок
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=18001)