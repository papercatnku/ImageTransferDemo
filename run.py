import sys,os
import time

import cv2
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt

from utils import StyleTransfer

style_model_path = './models/weights/'
style_img_path = './models/style/'

def init_valid_model_list():
    style_candidates = os.listdir(style_img_path)
    valid_style_list= [_name[:_name.rfind('.')] for _name in style_candidates]
    return tuple(valid_style_list)

use_cuda = 0
gpuid = 0
style_names = init_valid_model_list()

class Thread(QThread):
    change_pixmap_x = pyqtSignal(QPixmap)
    change_pixmap_y = pyqtSignal(QPixmap)
    change_pixmap_inf = pyqtSignal(str)
    change_pixmap_dis = pyqtSignal(str)

    def __init__(self, video_width, video_height, parent=None):
        QThread.__init__(self, parent=parent)
        self.video_height = video_height
        self.video_width = video_width
        self.stnet = StyleTransfer(gpuid, video_width, video_height)

    def transfer(self, x_np):
        return self.stnet.transfer(x_np)

    def change_model(self, model_tag):
        self.stnet.change_model(
            os.path.join(style_model_path,model_tag + '.json'),
            os.path.join(style_model_path,model_tag + '.params'))
        return

    def run(self):
        cap = cv2.VideoCapture(0)
        fps_update_cnt = 0
        fps_update_num = 10
        while True:
            display_time = time.time()
            ret, x_np = cap.read()
            if not ret:
                print('can not read camera')
                break
            x_np = cv2.cvtColor(x_np, cv2.COLOR_BGR2RGB)
            # Model process here
            y_np, inference_time = self.transfer(x_np)
            x_qt = QImage(x_np.data, x_np.shape[1], x_np.shape[0], QImage.Format_RGB888)
            x_qt = QPixmap.fromImage(x_qt)
            x_qt = x_qt.scaled(self.video_width, self.video_height, Qt.KeepAspectRatio)

            y_qt = QImage(y_np.data, y_np.shape[1], y_np.shape[0], QImage.Format_RGB888)
            y_qt = QPixmap.fromImage(y_qt)
            y_qt = y_qt.scaled(self.video_width, self.video_height, Qt.KeepAspectRatio)
            self.change_pixmap_x.emit(x_qt)
            self.change_pixmap_y.emit(y_qt)

            fps_update_cnt = (fps_update_cnt + 1) % fps_update_num
            if fps_update_cnt == 0:
                self.change_pixmap_inf.emit('    Infrence FPS: {0:.2f}'.format(
                    1 / inference_time if inference_time is not None else 0))
                display_time = time.time() - display_time
                self.change_pixmap_dis.emit('    Display FPS: {0:.2f}'.format(1 / display_time))


class StyleLabel(QLabel):
    signal = pyqtSignal(['QString'])
    def __init__(self, label, parent=None):
        QLabel.__init__(self, parent=parent)
        self.model_tag = style_names[label]

    def mousePressEvent(self, event):
        self.signal.emit(self.model_tag)

class GUI(QWidget):
    def __init__(
            self,
            video_width=640,
            video_height=480,
            padding=20,
            margin=100):
        super().__init__()
        self.title = 'ReCoNet Demo'

        self.video_width, self.video_height, self.padding, self.margin = video_width, video_height, padding, margin
        self.width = self.video_width * 2 + self.padding * 3

        style_num = len(style_names)
        self.style_size = (self.width - self.padding * (style_num + 1)) // style_num
        self.height = self.video_height + self.style_size + self.padding * 3
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.margin, self.margin, self.width, self.height)

        label_x = QLabel(self)
        label_x.move(self.padding, self.padding)
        label_x.resize(self.video_width, self.video_height)
        label_y = QLabel(self)
        label_y.move(self.video_width + self.padding * 2, self.padding)
        label_y.resize(self.video_width, self.video_height)

        label_inf = QLabel(self)
        label_inf.move(self.video_width + self.padding * 2, self.padding)
        label_inf.resize(150, 50)
        label_inf.setStyleSheet('background: yellow')
        label_dis = QLabel(self)
        label_dis.move(self.padding, self.padding)
        label_dis.resize(150, 50)
        label_dis.setStyleSheet('background: yellow')

        th = Thread(
            self.video_width,
            self.video_height,
            parent=self)
        th.change_pixmap_x.connect(label_x.setPixmap)
        th.change_pixmap_y.connect(label_y.setPixmap)
        th.change_pixmap_inf.connect(label_inf.setText)
        th.change_pixmap_dis.connect(label_dis.setText)

        label_style = []
        for i in range(len(style_names)):
            tmp_label = StyleLabel(i, self)
            tmp_label.move((self.style_size + self.padding) * i + self.padding, self.video_height + self.padding * 2)
            tmp_label.resize(self.style_size, self.style_size)
            tmp_pixmap = QPixmap('models/style/' + style_names[i] + '.jpg').scaled(self.style_size, self.style_size)
            tmp_label.setPixmap(tmp_pixmap)
            tmp_label.signal.connect(th.change_model)
            label_style.append(tmp_label)
        th.start()

        self.show()

def main():
    app = QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
