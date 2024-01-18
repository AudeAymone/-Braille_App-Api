from flask import Flask, request, send_file
import os 

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the request contains a file
    if 'image' not in request.files:
        return 'No image file in the request', 400

    print("j'ai été appelée")
    image_file = request.files['image']

    # Check if the file is empty
    if image_file.filename == '':
        return 'Empty file name', 400

    save_path = os.path.join(os.getcwd(), 'image', 'image.png')
    image_file.save(save_path)
    
    imageTraitee = os.path.join(os.getcwd(), 'image', 'imageTraitee.png')

    return send_file(imageTraitee)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 3000)