# sales_forecast
An interactive, AI-powered sales forecasting tool built with React (frontend) and Flask (backend). This application allows users to upload historical retail sales data (CSV), trains an XGBoost regression modell, and enables:

📊 Visualization of model performance (RMSE, MAPE)

📈 Store-wise analytics and statistics

🔮 Single-week predictions using economic factors

📅 Multi-week future sales forecasting for selected stores

Designed to help businesses make data-driven decisions based on historical and economic trends.

🚀 Features
Upload CSV sales data and auto-train a forecasting model

View store-wise metrics (sales, records, date range)

Generate real-time predictions with input parameters:

Temperature

Fuel price

CPI

Unemployment

Holiday indicator

Multi-week forecast with line charts

Notification system and loading animations

Responsive, modern UI with Tailwind CSS

🛠️ Tech Stack
Frontend	Backend	ML Model	Data Handling
React + Tailwind CSS	Flask + Flask-CORS	XGBoost Regressor	Pandas, NumPy

📁 Project Structure
/src – React frontend (UI, charts, forms)

salesflask.py – Flask backend with ML model and API routes

temp_data.csv – Temporary uploaded data

/forecast – Forecast endpoint logic

🧪 How to Run
Backend:

bash
Copy
Edit
python salesflask.py
Frontend:

bash
Copy
Edit
npm install
npm start
Make sure the backend is running on http://localhost:5000 to connect properly.
