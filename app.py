from flask import Flask , request , jsonify
from model import getPrediction

app = Flask(__name__)

@app.route("/predict",methods=['POST'])
def prediction():
    image = request.files.get("alphabet")
    prediction = getPrediction(image)
    return jsonify({
        "data":prediction,
        "message":"success"
    })

if __name__ =='__main__':
    app.run(debug=True)