🚗 Car Price Prediction: Linear Regression vs. Random Forest
Hey there! Welcome to my machine learning project where I set out to answer a simple question: Can vehicle specs alone accurately predict a car's price?

To test this, I built an end-to-end ML pipeline comparing a straightforward Linear Regression model against a Random Forest Regressor using a dataset of over 7,800 real-world cars.

Spoiler alert: The tree-based model blew linear regression out of the water. 🌲

💡 What's Inside This Project?
Car pricing isn't linear. A 50 horsepower boost on a brand-new luxury SUV changes the price tag completely differently than on a 15-year-old compact sedan.

I wanted to see how well traditional regression handles these complex, non-linear interactions compared to an ensemble tree model.

Key Learnings & Methodology:
Leak-Proof Preprocessing: Deduplication and filtering missing/zero prices were done before splitting the data so no test set information leaked into training.

Fixing Price Skewness: Car prices are heavily right-skewed (supercars skew the average big time). I applied a log1p(MSRP) transformation so the models penalize errors fairly across both $15k commuters and $200k exotics.

Clean Scikit-Learn Pipelines: Built modular Pipeline and ColumnTransformer setups to handle median imputation, standard scaling, and one-hot encoding without manual messiness.

Cross-Validation: Ran 5-fold CV to ensure performance numbers weren't just a lucky train/test split.

📊 The Results
Switching from Linear Regression to Random Forest dropped the average price prediction error by over 70%.

Metric	Linear Regression	Random Forest
R 
2
  Score ($)	0.814	0.985
Mean Absolute Error (MAE)	$10,370	$3,186
Root Mean Squared Error (RMSE)	$21,470	$6,204
🔑 What Drives Car Prices?
According to the Random Forest model:

Vehicle Year (~70% importance): Depreciation and tech updates carry the heaviest weight.

Engine HP (~20% importance): Power remains the single biggest mechanical feature driving MSRP.

(Note: While 0.985 
R 
2
  looks amazing, in the real world, near-identical trim levels of the same model year might end up in both train and test splits, making the score slightly optimistic!)

🛠️ Project Structure
Plaintext
├── car_price_model.ipynb        # The main Jupyter Notebook with all code & commentary
├── Car Dataset.csv              # Kaggle's Car Features & MSRP Dataset
├── pred_vs_actual.png           # Scatter plot comparing predictions against real prices
├── feature_importance.png       # Feature importance chart from the Random Forest model
├── .gitignore                   # Files Git should ignore
├── LICENSE                      # MIT License
└── README.md                    # You are here!
🚀 Running It Yourself
Want to test a sample prediction or play with the code?

Clone the repo:

Bash
git clone https://github.com/rs-rohitsingh-official/car-price-prediction.git
cd car-price-prediction
Install the dependencies:

Bash
pip install numpy pandas matplotlib scikit-learn joblib
Launch the notebook:

Bash
jupyter notebook car_price_model.ipynb
💬 Thoughts & Next Steps
If I take this project further, my next steps would be:

Grouped Train/Test Splits: Splitting data by specific car models (rather than random rows) to see how well it predicts on completely unseen vehicle lines.

Gradient Boosting: Testing XGBoost or LightGBM to see if we can shave off a few more dollars from the MAE.

Got feedback or ideas on how to improve this? Feel free to open an issue, submit a PR, or reach out to connect on LinkedIn!
