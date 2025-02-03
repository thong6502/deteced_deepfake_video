import os
from yt_dlp import YoutubeDL

# Tạo thư mục lưu video
DOWNLOAD_DIR = "videos"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_video(url):
    """
    Tải video từ URL và lưu vào thư mục downloads.
    Trả về đường dẫn file và tên file.
    """
    ydl_opts = {
        #'format': 'bestvideo[height<=720][ext=mp4]/best[ext=mp4]',  # Ưu tiên video 1080p MP4 với âm thanh
        'format': 'worst',
        #'merge_output_format': None,  # Không dùng FFmpeg
        'outtmpl': f'{DOWNLOAD_DIR}/%(id)s.%(ext)s',  # Đường dẫn lưu file
        'windowsfilenames': True,  # Tự động thay thế ký tự không hợp lệ trên Windows
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)  # Đường dẫn file đầy đủ
        file_name = os.path.basename(file_path)  # Lấy tên file từ đường dẫn

    return file_path, file_name
