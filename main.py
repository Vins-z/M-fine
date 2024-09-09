import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QTextEdit, QFileDialog
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SalesAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sales Data Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_data_tab()
        self.create_correlation_tab()
        self.create_model_tab()
        self.create_results_tab()

    def create_data_tab(self):
        data_tab = QWidget()
        layout = QVBoxLayout()
        data_tab.setLayout(layout)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.data_info = QTextEdit()
        self.data_info.setReadOnly(True)
        layout.addWidget(self.data_info)

        self.tab_widget.addTab(data_tab, "Data")

    def create_correlation_tab(self):
        correlation_tab = QWidget()
        layout = QVBoxLayout()
        correlation_tab.setLayout(layout)

        self.correlation_canvas = FigureCanvas(plt.Figure(figsize=(10, 8)))
        layout.addWidget(self.correlation_canvas)

        self.tab_widget.addTab(correlation_tab, "Correlation")

    def create_model_tab(self):
        model_tab = QWidget()
        layout = QVBoxLayout()
        model_tab.setLayout(layout)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        layout.addWidget(self.model_info)

        self.loss_canvas = FigureCanvas(plt.Figure(figsize=(10, 6)))
        layout.addWidget(self.loss_canvas)

        self.tab_widget.addTab(model_tab, "Model")

    def create_results_tab(self):
        results_tab = QWidget()
        layout = QVBoxLayout()
        results_tab.setLayout(layout)

        self.results_canvas = FigureCanvas(plt.Figure(figsize=(10, 6)))
        layout.addWidget(self.results_canvas)

        self.tab_widget.addTab(results_tab, "Results")

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.advertising_df = pd.read_csv(file_name)
            self.preprocess_data()
            self.update_data_info()
            self.plot_correlation()

    def preprocess_data(self):
        # Process the date column
        self.advertising_df['ORDERDATE'] = pd.to_datetime(self.advertising_df['ORDERDATE'], errors='coerce')
        self.advertising_df['YEAR'] = self.advertising_df['ORDERDATE'].dt.year
        self.advertising_df['MONTH'] = self.advertising_df['ORDERDATE'].dt.month
        self.advertising_df['DAY'] = self.advertising_df['ORDERDATE'].dt.day
        self.advertising_df.drop(columns=['ORDERDATE'], inplace=True)

        # Encoding categorical columns
        label_encoder = LabelEncoder()
        categorical_columns = ['CITY', 'COUNTRY', 'PRODUCTLINE', 'CUSTOMERNAME', 'DEALSIZE', 'OCCUPATION']
        for col in categorical_columns:
            self.advertising_df[col] = label_encoder.fit_transform(self.advertising_df[col])

    def update_data_info(self):
        info = f"Dataset shape: {self.advertising_df.shape}\n\n"
        info += "Data Info:\n" + self.advertising_df.info().__str__() + "\n\n"
        info += "Data Description:\n" + self.advertising_df.describe().__str__()
        self.data_info.setText(info)

    def plot_correlation(self):
        numeric_columns = self.advertising_df.select_dtypes(include=[np.number]).columns
        correlation = self.advertising_df[numeric_columns].corr()

        ax = self.correlation_canvas.figure.subplots()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        self.correlation_canvas.draw()

    def train_model(self):
        feature_columns = ['PRICE', 'CITY', 'COUNTRY', 'PRODUCTLINE', 'CUSTOMERNAME', 'DEALSIZE', 'YEAR', 'MONTH', 'DAY', 'OCCUPATION', 'ANNUAL_INCOME']
        X = self.advertising_df[feature_columns]
        y = self.advertising_df['SALES']

        normalized_feature = keras.utils.normalize(X.values)
        X_train, X_test, y_train, y_test = train_test_split(normalized_feature, y, test_size=0.3, random_state=42)

        model = Sequential()
        model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=0)

        self.plot_loss(history)
        self.plot_results(model, X_test, y_test)

        train_mse = model.evaluate(X_train, y_train, verbose=0)[1]
        test_mse = model.evaluate(X_test, y_test, verbose=0)[1]
        self.model_info.setText(f"Train MSE: {train_mse:.4f}\nTest MSE: {test_mse:.4f}")

    def plot_loss(self, history):
        ax = self.loss_canvas.figure.subplots()
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss (MSE) on Training and Validation Data')
        ax.set_ylabel('Loss (Mean Squared Error)')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        self.loss_canvas.draw()

    def plot_results(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        ax = self.results_canvas.figure.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title('Actual vs Predicted Sales')
        self.results_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = SalesAnalysisApp()
    main_window.show()
    sys.exit(app.exec_())