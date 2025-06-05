import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from src.views.main_window import App
from src.ultis.get_resource_path import ResourcePath

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = App()
    window.setWindowTitle("Plant Disease Prediction")
    window.setWindowIcon(QIcon(ResourcePath.LOGO_PATH))

    window.show()
    sys.exit(app.exec_())
