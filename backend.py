import os
from flask import Flask, request, jsonify, render_template

from src.downloader import download_video

app = Flask(__name__)
@app.route('/')
# def home():
#   return "Hello, World!"
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
  file_path, file_name = download_video(url)
  return jsonify({
    'file_path': file_path,
    'file_name': file_name
  })
if __name__ == '__main__':
  app.run(debug=True, port=5000)