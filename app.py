from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Models
reg_model = pickle.load(open("regression_model.pkl", "rb"))
clf_model = pickle.load(open("classification_model.pkl", "rb"))

# Home Route
@app.route('/')
def home():
    return render_template('index.html')  # your existing homepage

# Results Route
@app.route('/results', methods=['POST'])
def results():
    # Get slider inputs from form
    work_balance = float(request.form['work_balance'])
    culture = float(request.form['culture'])
    career = float(request.form['career'])
    comp = float(request.form['comp'])
    management = float(request.form['management'])
    
    # Optional extra questions (priority / entry-level)
    priority = request.form.get('priority', 'default')
    entry_level = request.form.get('entry_level', 'default')
    
    # Prepare features for models
    features = np.array([[work_balance, culture, career, comp, management]])
    
    # Regression prediction (overall rating)
    overall_pred = reg_model.predict(features)[0]
    
    # Classification prediction (best company)
    company_pred = clf_model.predict(features)[0]
    
    # Pass visuals to template
    regression_chart = "regression_importance.png"
    pros_chart = "pros_sentiment.png"
    cons_chart = "cons_sentiment.png"
    
    # Render Results Page
    return render_template(
        'result.html',
        overall_rating=round(overall_pred, 2),
        company=company_pred,
        regression_chart=regression_chart,
        pros_chart=pros_chart,
        cons_chart=cons_chart,
        priority=priority,
        entry_level=entry_level
    )

# Run App
if __name__ == "__main__":
    app.run(debug=True)
