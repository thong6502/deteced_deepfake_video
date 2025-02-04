import os
from flask import Flask, request, jsonify, render_template

from src.downloader import download_video
from src.utils import load_model, process

app = Flask(__name__)
@app.route('/')
def home():
  try:
    #Flask mặc định tìm kiếm file index.html trong thư mục templates

    #Kiểm tra xem file index.html có tồn tại không
    if not os.path.exists('templates/index1.html'):
      raise Exception("Template file 'index.html' not found")
    return render_template('index.html')
  except Exception as e:
    app.logger.error(f"{e}")
    return "Error", 500

@app.route('/download', methods=['POST'])
def download():
  url = request.json['url']
  file_path, _ = download_video(url)
  return file_path

@app.route('/process', methods=['POST'])
def process_video():
  file_path = download()
  model = load_model("trained_model/best.pt")
  output_path = process(file_path, model, "videos/output")
  return jsonify({
    'output_path': f"http://localhost:5000/{output_path}"
  })
if __name__ == '__main__':
  app.run(debug=True, port=5000)