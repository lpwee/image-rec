import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/image", methods=["POST"])
def image():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no image provided"}), 400
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400
    return jsonify({"status": "received", "shape": frame.shape})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6767, debug=True)
