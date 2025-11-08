# Bike Sales Analysis - India

A comprehensive data analysis project for motorcycle resale value prediction and strategic insights in the Indian market.

## 📊 Overview

This project analyzes bike sales data across India to provide insights into motorcycle resale values, price retention factors, and market trends. It includes statistical analysis, machine learning models, visualization tools, and answers to specific business questions related to motorcycle resale value consulting.

## 🎯 Key Features

- **Correlation Analysis**: Evaluate relationships between mileage, engine capacity, and price retention
- **Machine Learning Models**: Compare 9 different regression models for resale price prediction
- **Anomaly Detection**: Identify fraudulent mileage claims using statistical methods
- **Geographic Analysis**: Calculate distances using Haversine formula for evacuation planning
- **Interactive Visualizations**: Generate plots, heatmaps, and interactive HTML charts
- **Streamlit Dashboard**: Interactive web interface for exploring question solutions

## 📁 Dataset

**File**: `bike_sales_india.csv`

The dataset contains motorcycle sales records with the following features:
- State, City Tier (Metro, Tier 1, Tier 2, Tier 3)
- Brand and Model
- Original Price and Resale Price (INR)
- Year of Manufacture and Registration Year
- Engine Capacity (cc)
- Fuel Type (Petrol, Electric, Hybrid)
- Mileage (km/l)
- Average Daily Distance (km)
- Owner Type (First, Second, Third)
- Insurance Status (Active, Expired, Not Available)
- Seller Type (Individual, Dealer)

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Dependencies

Install all required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn plotly scipy haversine scikit-learn streamlit
```

Or create a requirements file:

```txt
pandas
numpy
matplotlib
seaborn
plotly
scipy
haversine
scikit-learn
streamlit
```

Then install:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Main Analysis Script

Run the comprehensive analysis:

```bash
python bike_analysis.py
```

This script will:
- Load and preprocess the dataset
- Generate descriptive statistics
- Create visualizations (correlation heatmaps, histograms, boxplots)
- Train and compare 9 ML models
- Answer all 9 business questions
- Save results to CSV files and images

**Note**: Update the file paths in the script (lines 19, 26) to match your local directory structure:
```python
output_folder = r"YOUR_OUTPUT_PATH"
df = pd.read_csv(r"YOUR_PATH/bike_sales_india.csv")
```

### 2. Interactive Streamlit Dashboard

Launch the interactive web interface:

```bash
streamlit run bike_vicky.py
```

The dashboard allows you to:
- Select different questions from the sidebar
- View calculation details for each question
- Explore answers interactively

## 📝 Business Questions Answered

The project addresses 9 specific analytical questions:

1. **Q1**: Correlation analysis for TVS motorcycles in Gujarat
2. **Q2**: Correlation analysis for Yamaha motorcycles in West Bengal
3. **Q3**: Predicted resale price for a specific Tier 2 motorcycle
4. **Q4**: Forecast 2027 resale value for Kawasaki Versys 650 in Delhi
5. **Q5**: Fraudulent mileage detection for Electric vehicles in Maharashtra
6. **Q6**: Fraudulent mileage detection for Electric vehicles in Uttar Pradesh
7. **Q7**: Haversine distance calculation for evacuation planning
8. **Q8**: Identify closest community to Central Command Post
9. **Q9**: Optimal evacuation route using nearest neighbor strategy

## 🤖 Machine Learning Models

The project compares the following regression models for resale price prediction:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

Models are evaluated using:
- Mean Squared Error (MSE)
- R² Score
- Cross-validation

## 📈 Output Files

After running `bike_analysis.py`, the following files are generated:

### CSV Files
- `descriptive_stats.csv` - Statistical summary of the dataset
- `correlation_matrix.csv` - Correlation coefficients between numeric features
- `model_comparison.csv` - Performance metrics for all ML models
- `question_answer_solution.csv` - Answers to all 9 business questions
- `brand_summary.csv`, `state_summary.csv` - Aggregated statistics

### Visualizations
- `correlation_heatmap.png` - Correlation matrix visualization
- `histograms.png` - Distribution of numeric features
- `boxplots.png` - Box plots for outlier detection
- `pairplot.png` - Pairwise relationships between features
- `numeric_distributions.png` - Histograms of all numeric columns
- `price_retention_by_brand.png` - Brand-wise price retention analysis
- `fuel_type_distribution.png` - Fuel type distribution

### Interactive HTML Files
- `interactive_scatter.html` - Interactive scatter plot
- `mileage_vs_retention_interactive.html` - Mileage vs retention analysis
- `price_by_brand_interactive.html` - Price distribution by brand

## 📊 Project Structure

```
ga6/
├── README.md                          # This file
├── bike_analysis.py                   # Main analysis script
├── bike_vicky.py                      # Streamlit dashboard
├── bike_sales_india.csv               # Dataset
├── bike_question.txt                  # Business questions
├── answer.csv                         # Question answers
├── pdf/                               # Additional resources
├── __pycache__/                       # Python cache
└── [output files]                     # Generated visualizations and CSVs
```

## 🧪 Key Analytical Techniques

- **Pearson Correlation**: Measure linear relationships between variables
- **IQR-based Anomaly Detection**: Identify outliers using interquartile range
- **Haversine Formula**: Calculate geographic distances
- **Nearest Neighbor Algorithm**: Optimize evacuation routes
- **Time Series Forecasting**: Predict future resale values
- **Feature Scaling**: StandardScaler for improved model performance

## 🎓 Use Cases

This project is valuable for:
- **Motorcycle Dealerships**: Understand resale value factors
- **Market Researchers**: Analyze trends across different regions
- **Investment Analysts**: Forecast future motorcycle values
- **Quality Assurance Teams**: Detect fraudulent mileage claims
- **Students & Data Scientists**: Learn practical data analysis techniques

## 📄 License

This is an educational/analytical project. Please ensure you have appropriate rights to use the dataset.

## 🤝 Contributing

This appears to be a graded assignment (GA6). If you're using this for learning:
1. Understand the code thoroughly
2. Modify paths to match your environment
3. Experiment with different models and parameters
4. Try answering additional business questions

## ⚠️ Important Notes

1. **File Paths**: Update hardcoded paths in `bike_analysis.py` before running
2. **Output Directory**: Ensure the output directory exists or the script will create it
3. **Data Quality**: The script handles missing data and performs basic preprocessing
4. **Computation Time**: Model training may take several minutes depending on your hardware

## 📞 Support

For issues or questions related to:
- The code implementation
- Understanding the analysis
- Modifying for your use case

Review the code comments and function docstrings for detailed explanations.

---

**Last Updated**: 2024  
**Python Version**: 3.7+  
**Status**: Active
