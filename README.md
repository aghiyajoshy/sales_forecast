# 📈 Sales Forecasting App

An AI-powered retail analytics platform for predicting and forecasting sales using machine learning. Built with React.js frontend and Flask backend, this application helps retailers make data-driven decisions by analyzing historical sales data and generating accurate forecasts.

## ✨ Features

- **📊 Interactive Dashboard** - Real-time model performance metrics and store analytics
- **🤖 AI-Powered Predictions** - Generate single sales predictions based on economic factors
- **📅 Multi-Period Forecasting** - Create detailed sales forecasts for up to 52 weeks
- **📁 CSV Data Upload** - Easy data import with automatic model training
- **📈 Data Visualization** - Interactive charts and graphs using Recharts
- **🏪 Multi-Store Support** - Analyze and forecast for multiple retail locations
- **🎯 Real-time Analytics** - Live model performance tracking and validation

## 🛠️ Tech Stack

**Frontend:**
- React.js 18+
- Recharts (Data Visualization)
- Lucide React (Icons)

**Backend:**
- Python Flask
- Scikit-learn / TensorFlow (ML Models)
- Pandas (Data Processing)
- NumPy (Numerical Computing)

## 🚀 Quick Start

### Prerequisites
- Node.js 14+ and npm
- Python 3.8+
- Git

### Frontend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/sales-forecasting-app.git
cd sales-forecasting-app

# Install dependencies
npm install

# Install required packages
npm install recharts lucide-react


# Start development server
npm start
```

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install flask flask-cors pandas numpy scikit-learn

# Start Flask server
python app.py
```

The application will be available at `http://localhost:3000`

## 📋 Usage

1. **Upload Data**: Start by uploading your historical sales CSV file
2. **View Dashboard**: Monitor model performance and store statistics
3. **Generate Predictions**: Input economic factors to predict sales
4. **Create Forecasts**: Generate multi-week sales forecasts for planning
5. **Analyze Results**: Use interactive charts to understand trends

### Required CSV Format
Your sales data should include columns for:
- Store ID
- Date
- Weekly Sales
- Holiday Flag
- Temperature
- Fuel Price
- CPI (Consumer Price Index)
- Unemployment Rate

## 📊 Model Performance

The application uses advanced machine learning algorithms to achieve:
- Low RMSE (Root Mean Square Error) for accurate predictions
- MAPE (Mean Absolute Percentage Error) tracking
- Cross-validation for model reliability
- Real-time performance monitoring

## 🎯 Key Metrics Tracked

- **Model RMSE**: Prediction accuracy measurement
- **Model MAPE**: Percentage error tracking
- **Total Stores**: Multi-location support
- **Training Records**: Data volume processed
- **Forecast Accuracy**: Long-term prediction reliability

## 🔧 Configuration

### Environment Variables
```env
REACT_APP_API_BASE_URL=http://localhost:5000
FLASK_PORT=5000
FLASK_DEBUG=True
```

### API Endpoints
- `GET /` - API status and health check
- `POST /upload` - Upload CSV and train model
- `GET /stores` - Retrieve store information
- `GET /model_info` - Get model performance metrics
- `POST /predict` - Generate single prediction
- `POST /forecast` - Create multi-period forecast

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 Future Enhancements

- [ ] Advanced ML models (LSTM, Prophet)
- [ ] Real-time data integration
- [ ] Mobile responsive design improvements
- [ ] Export functionality for forecasts
- [ ] Advanced filtering and segmentation
- [ ] Automated report generation
- [ ] Integration with retail POS systems

## 🐛 Known Issues

- Large CSV files (>100MB) may take longer to process
- Internet connection required for initial setup
- Browser compatibility: Chrome, Firefox, Safari, Edge

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Create React App
- Powered by Recharts for data visualization
- Icons provided by Lucide React
- Styling with Tailwind CSS


**⭐ Star this repository if you find it helpful!**

Made with ❤️ for retail analytics and data-driven decision making.
