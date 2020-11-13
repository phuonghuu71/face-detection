# face-detection
- Chương trình chạy trên Python3
- Các thư viện cần tải:
  * numpy phiên bản 1.19.3
  * opencv-python
  * opencv-contrib-python
  * pillow
- Tải các thư viện bằng pip:
  * pip install numpy==1.19.3
  * pip install opencv-python
  * pip install opencv-contrib-python
  * pip install pillow
- Chạy chương trình bằng lệnh python3 -u example.py
- Nếu gặp lỗi error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor': chưa xác định được camera
- Nếu gặp lỗi AttributeError: module 'cv2.cv2' has no attribute 'face': xóa thư viện opencv-contrib-python và tải lại
  * Lệnh xóa thư viện: pip uninstall opencv-contrib-python
  * Lệnh tải thư viện: pip install opencv-contrib-python
- Nếu gặp lỗi ImportError: numpy.core.multiarray failed to import: phiên bản numpy chưa đúng
- Nếu gặp lỗi import cv2ImportError: No module named cv2: chưa tải opencv-python
