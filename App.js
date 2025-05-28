import React, { useState, useEffect } from 'react';
import { Upload, TrendingUp, BarChart3, Store, Calendar, DollarSign, AlertCircle, CheckCircle } from 'lucide-react';

const API_BASE_URL = 'http://localhost:5000';

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
  },
  header: {
    background: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(10px)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
    padding: '1rem 2rem',
    color: 'white'
  },
  headerContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    alignItems: 'center',
    gap: '1rem'
  },
  title: {
    fontSize: '1.5rem',
    fontWeight: '700',
    margin: 0
  },
  main: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '2rem',
    display: 'grid',
    gap: '2rem'
  },
  card: {
    background: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '16px',
    padding: '2rem',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    border: '1px solid rgba(255, 255, 255, 0.3)'
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    marginBottom: '1.5rem'
  },
  cardTitle: {
    fontSize: '1.25rem',
    fontWeight: '600',
    margin: 0,
    color: '#1f2937'
  },
  uploadArea: {
    border: '2px dashed #d1d5db',
    borderRadius: '12px',
    padding: '3rem 2rem',
    textAlign: 'center',
    backgroundColor: '#f9fafb',
    transition: 'all 0.2s ease',
    cursor: 'pointer'
  },
  uploadAreaActive: {
    borderColor: '#667eea',
    backgroundColor: '#f0f4ff'
  },
  button: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    padding: '0.75rem 1.5rem',
    fontSize: '0.875rem',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  buttonSecondary: {
    background: 'transparent',
    color: '#667eea',
    border: '2px solid #667eea'
  },
  input: {
    width: '100%',
    padding: '0.75rem',
    border: '2px solid #e5e7eb',
    borderRadius: '8px',
    fontSize: '0.875rem',
    transition: 'border-color 0.2s ease',
    outline: 'none'
  },
  inputFocus: {
    borderColor: '#667eea'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1rem'
  },
  gridTwo: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
    gap: '2rem'
  },
  formGroup: {
    marginBottom: '1rem'
  },
  label: {
    display: 'block',
    fontSize: '0.875rem',
    fontWeight: '500',
    color: '#374151',
    marginBottom: '0.5rem'
  },
  alert: {
    padding: '1rem',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    fontSize: '0.875rem',
    marginBottom: '1rem'
  },
  alertError: {
    backgroundColor: '#fef2f2',
    color: '#dc2626',
    border: '1px solid #fecaca'
  },
  alertSuccess: {
    backgroundColor: '#f0fdf4',
    color: '#16a34a',
    border: '1px solid #bbf7d0'
  },
  metric: {
    background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
    padding: '1.5rem',
    borderRadius: '12px',
    textAlign: 'center'
  },
  metricValue: {
    fontSize: '2rem',
    fontWeight: '700',
    color: '#1e293b',
    margin: '0.5rem 0'
  },
  metricLabel: {
    fontSize: '0.875rem',
    color: '#64748b',
    fontWeight: '500'
  },
  forecastResult: {
    background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
    padding: '1rem',
    borderRadius: '8px',
    marginTop: '1rem'
  },
  loading: {
    display: 'inline-block',
    width: '20px',
    height: '20px',
    border: '3px solid #f3f3f3',
    borderTop: '3px solid #667eea',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  }
};

// Add CSS animation for loading spinner
const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
  .card:hover {
    transform: translateY(-4px);
    transition: transform 0.2s ease;
  }
`;
document.head.appendChild(styleSheet);

const SalesForecastingDashboard = () => {
  const [file, setFile] = useState(null);
  const [modelTrained, setModelTrained] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [stores, setStores] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [forecast, setForecast] = useState(null);
  
  // Prediction form state
  const [predictionForm, setPredictionForm] = useState({
    holiday_flag: 0,
    temperature: 70,
    fuel_price: 3.5,
    cpi: 200,
    unemployment: 7
  });
  
  // Forecast form state
  const [forecastForm, setForecastForm] = useState({
    store_id: 1,
    periods: 4,
    holiday_flag: 0,
    temperature: 70,
    fuel_price: 3.5,
    cpi: 200,
    unemployment: 7
  });

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/`);
      const data = await response.json();
      setModelTrained(data.model_trained);
      
      if (data.model_trained) {
        await loadModelInfo();
        await loadStores();
      }
    } catch (err) {
      console.error('Error checking model status:', err);
    }
  };

  const loadModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model_info`);
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      console.error('Error loading model info:', err);
    }
  };

  const loadStores = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stores`);
      const data = await response.json();
      setStores(data.stores || []);
    } catch (err) {
      console.error('Error loading stores:', err);
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess('Model trained successfully!');
        setModelTrained(true);
        await loadModelInfo();
        await loadStores();
      } else {
        setError(data.error || 'Upload failed');
      }
    } catch (err) {
      setError('Network error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handlePrediction = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(predictionForm)
      });

      const data = await response.json();

      if (response.ok) {
        setPrediction(data);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError('Network error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleForecast = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/forecast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(forecastForm)
      });

      const data = await response.json();

      if (response.ok) {
        setForecast(data);
      } else {
        setError(data.error || 'Forecast failed');
      }
    } catch (err) {
      setError('Network error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <TrendingUp size={24} />
          <h1 style={styles.title}>Sales Forecasting Dashboard</h1>
        </div>
      </header>

      <main style={styles.main}>
        {error && (
          <div style={{...styles.alert, ...styles.alertError}}>
            <AlertCircle size={20} />
            {error}
          </div>
        )}

        {success && (
          <div style={{...styles.alert, ...styles.alertSuccess}}>
            <CheckCircle size={20} />
            {success}
          </div>
        )}

        {/* File Upload Section */}
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <Upload size={24} color="#667eea" />
            <h2 style={styles.cardTitle}>Upload Sales Data</h2>
          </div>
          
          <div 
            style={{
              ...styles.uploadArea,
              ...(file ? styles.uploadAreaActive : {})
            }}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <Upload size={48} color="#9ca3af" style={{margin: '0 auto 1rem'}} />
            <p style={{color: '#6b7280', margin: '0 0 1rem 0'}}>
              {file ? file.name : 'Click to upload CSV file or drag and drop'}
            </p>
            <input
              id="fileInput"
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
              style={{display: 'none'}}
            />
          </div>
          
          <button 
            style={styles.button} 
            onClick={handleFileUpload}
            disabled={loading || !file}
          >
            {loading ? <div style={styles.loading}></div> : <Upload size={16} />}
            {loading ? 'Training Model...' : 'Upload & Train Model'}
          </button>
        </div>

        {/* Model Info Section */}
        {modelTrained && modelInfo && (
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <BarChart3 size={24} color="#667eea" />
              <h2 style={styles.cardTitle}>Model Performance</h2>
            </div>
            
            <div style={styles.grid}>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Test RMSE</div>
                <div style={styles.metricValue}>
                  {modelInfo.performance_metrics.test_rmse.toFixed(2)}
                </div>
              </div>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Test MAPE (%)</div>
                <div style={styles.metricValue}>
                  {modelInfo.performance_metrics.test_mape.toFixed(2)}%
                </div>
              </div>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Features Used</div>
                <div style={styles.metricValue}>
                  {modelInfo.features_count}
                </div>
              </div>
              <div style={styles.metric}>
                <div style={styles.metricLabel}>Training Samples</div>
                <div style={styles.metricValue}>
                  {modelInfo.performance_metrics.training_samples}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Prediction and Forecast Section */}
        {modelTrained && (
          <div style={styles.gridTwo}>
            {/* Single Prediction */}
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <DollarSign size={24} color="#667eea" />
                <h2 style={styles.cardTitle}>Single Prediction</h2>
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Holiday Flag</label>
                <select
                  style={styles.input}
                  value={predictionForm.holiday_flag}
                  onChange={(e) => setPredictionForm({...predictionForm, holiday_flag: parseInt(e.target.value)})}
                >
                  <option value={0}>No Holiday</option>
                  <option value={1}>Holiday</option>
                </select>
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Temperature (°F)</label>
                <input
                  type="number"
                  style={styles.input}
                  value={predictionForm.temperature}
                  onChange={(e) => setPredictionForm({...predictionForm, temperature: parseFloat(e.target.value)})}
                />
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Fuel Price ($)</label>
                <input
                  type="number"
                  step="0.01"
                  style={styles.input}
                  value={predictionForm.fuel_price}
                  onChange={(e) => setPredictionForm({...predictionForm, fuel_price: parseFloat(e.target.value)})}
                />
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>CPI</label>
                <input
                  type="number"
                  style={styles.input}
                  value={predictionForm.cpi}
                  onChange={(e) => setPredictionForm({...predictionForm, cpi: parseFloat(e.target.value)})}
                />
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Unemployment Rate (%)</label>
                <input
                  type="number"
                  step="0.1"
                  style={styles.input}
                  value={predictionForm.unemployment}
                  onChange={(e) => setPredictionForm({...predictionForm, unemployment: parseFloat(e.target.value)})}
                />
              </div>
              
              <button style={styles.button} onClick={handlePrediction} disabled={loading}>
                {loading ? <div style={styles.loading}></div> : <TrendingUp size={16} />}
                Predict Sales
              </button>
              
              {prediction && (
                <div style={styles.forecastResult}>
                  <h4 style={{margin: '0 0 0.5rem 0', color: '#92400e'}}>Predicted Weekly Sales</h4>
                  <p style={{fontSize: '1.5rem', fontWeight: '700', color: '#92400e', margin: 0}}>
                    ${prediction.prediction.toLocaleString()}
                  </p>
                </div>
              )}
            </div>

            {/* Forecast */}
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <Calendar size={24} color="#667eea" />
                <h2 style={styles.cardTitle}>Generate Forecast</h2>
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Store ID</label>
                <select
                  style={styles.input}
                  value={forecastForm.store_id}
                  onChange={(e) => setForecastForm({...forecastForm, store_id: parseInt(e.target.value)})}
                >
                  {stores.map(store => (
                    <option key={store.store_id} value={store.store_id}>
                      Store {store.store_id} (Avg: ${store.avg_weekly_sales.toLocaleString()})
                    </option>
                  ))}
                </select>
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Forecast Periods (weeks)</label>
                <input
                  type="number"
                  min="1"
                  max="12"
                  style={styles.input}
                  value={forecastForm.periods}
                  onChange={(e) => setForecastForm({...forecastForm, periods: parseInt(e.target.value)})}
                />
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Temperature (°F)</label>
                <input
                  type="number"
                  style={styles.input}
                  value={forecastForm.temperature}
                  onChange={(e) => setForecastForm({...forecastForm, temperature: parseFloat(e.target.value)})}
                />
              </div>
              
              <div style={styles.formGroup}>
                <label style={styles.label}>Fuel Price ($)</label>
                <input
                  type="number"
                  step="0.01"
                  style={styles.input}
                  value={forecastForm.fuel_price}
                  onChange={(e) => setForecastForm({...forecastForm, fuel_price: parseFloat(e.target.value)})}
                />
              </div>
              
              <button style={styles.button} onClick={handleForecast} disabled={loading}>
                {loading ? <div style={styles.loading}></div> : <Calendar size={16} />}
                Generate Forecast
              </button>
              
              {forecast && (
                <div style={styles.forecastResult}>
                  <h4 style={{margin: '0 0 1rem 0', color: '#92400e'}}>
                    Forecast for Store {forecast.store_id}
                  </h4>
                  {forecast.forecasts.map((item, index) => (
                    <div key={index} style={{
                      display: 'flex', 
                      justifyContent: 'space-between',
                      marginBottom: '0.5rem',
                      padding: '0.5rem',
                      background: 'rgba(255, 255, 255, 0.5)',
                      borderRadius: '4px'
                    }}>
                      <span>Week {item.week}</span>
                      <span style={{fontWeight: '600'}}>
                        ${item.predicted_sales.toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Stores Information */}
        {stores.length > 0 && (
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <Store size={24} color="#667eea" />
              <h2 style={styles.cardTitle}>Stores Overview</h2>
            </div>
            
            <div style={styles.grid}>
              {stores.slice(0, 6).map(store => (
                <div key={store.store_id} style={styles.metric}>
                  <div style={styles.metricLabel}>Store {store.store_id}</div>
                  <div style={styles.metricValue}>
                    ${store.avg_weekly_sales.toLocaleString()}
                  </div>
                  <div style={styles.metricLabel}>Avg Weekly Sales</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default SalesForecastingDashboard;