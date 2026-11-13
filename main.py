import os
import ast
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Base directory for reliable relative path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets ===========================================
sym_des = pd.read_csv(os.path.join(BASE_DIR, "datasets", "symptoms_df.csv"))
precautions = pd.read_csv(os.path.join(BASE_DIR, "datasets", "precautions_df.csv"))
workout = pd.read_csv(os.path.join(BASE_DIR, "datasets", "workout_df.csv"))
description = pd.read_csv(os.path.join(BASE_DIR, "datasets", "description.csv"))
medications = pd.read_csv(os.path.join(BASE_DIR, "datasets", "medications.csv"))
diets = pd.read_csv(os.path.join(BASE_DIR, "datasets", "diets.csv"))

# Load model ==============================================
svc = pickle.load(open(os.path.join(BASE_DIR, "models", "svc.pkl"), 'rb'))

# CRITICAL: Preserve exact canonical feature keys and ordering for ML model
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91,
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
    'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
    'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# CRITICAL: Preserve exact original disease classification mapping
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice',
    29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
    13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins',
    26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis',
    5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}

# =========================================================
# DATA NORMALIZATION & LOOKUP MAPS
# =========================================================

def clean_disease_key(s):
    """Normalize disease names for reliable cross-dataset lookups."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    aliases = {
        'peptic ulcer diseae': 'peptic ulcer disease',
        'diabetes ': 'diabetes',
        'hypertension ': 'hypertension',
        '(vertigo) paroymsal  positional vertigo': '(vertigo) paroymsal positional vertigo',
    }
    s = ' '.join(s.split())
    return aliases.get(s, s)

def display_disease_name(raw):
    """Generate professional, clean disease title for user display."""
    if not isinstance(raw, str):
        return "Unknown Condition"
    name = raw.strip()
    special_titles = {
        'Peptic ulcer diseae': 'Peptic Ulcer Disease',
        '(vertigo) Paroymsal  Positional Vertigo': 'Benign Paroxysmal Positional Vertigo (BPPV)',
        'hepatitis A': 'Hepatitis A',
        'GERD': 'GERD (Acid Reflux)',
        'AIDS': 'AIDS',
        'Diabetes ': 'Diabetes',
        'Hypertension ': 'Hypertension (High Blood Pressure)',
        'Dimorphic hemmorhoids(piles)': 'Hemorrhoids (Piles)',
        'Paralysis (brain hemorrhage)': 'Paralysis (Brain Hemorrhage)',
        'Common Cold': 'Common Cold',
    }
    if name in special_titles:
        return special_titles[name]
    return ' '.join(w.capitalize() if not w.isupper() else w for w in name.split())

def format_symptom_label(key):
    """Convert raw dataset keys into human-friendly medical symptom labels."""
    overrides = {
        'spotting_ urination': 'Spotting Urination',
        'foul_smell_of urine': 'Foul Smell of Urine',
        'toxic_look_(typhos)': 'Toxic Look (Typhos)',
        'dischromic _patches': 'Dischromic Patches',
        'fluid_overload.1': 'Fluid Overload (Systemic)',
        'cold_hands_and_feets': 'Cold Hands and Feet',
        'extra_marital_contacts': 'High-Risk Intimate Contact',
        'patches_in_throat': 'Patches in Throat',
        'loss_of_appetite': 'Loss of Appetite',
        'pain_behind_the_eyes': 'Pain Behind the Eyes',
        'weakness_in_limbs': 'Weakness in Limbs',
        'pain_during_bowel_movements': 'Pain During Bowel Movements',
        'pain_in_anal_region': 'Pain in Anal Region',
        'irritation_in_anus': 'Irritation in Anus',
        'puffy_face_and_eyes': 'Puffy Face and Eyes',
        'swollen_extremeties': 'Swollen Extremities',
        'drying_and_tingling_lips': 'Drying & Tingling Lips',
        'weakness_of_one_body_side': 'Weakness of One Body Side',
        'loss_of_smell': 'Loss of Smell',
        'continuous_feel_of_urine': 'Continuous Urge to Urinate',
        'passage_of_gases': 'Excessive Flatulence (Gas)',
        'red_spots_over_body': 'Red Spots Over Body',
        'history_of_alcohol_consumption': 'History of Alcohol Consumption',
        'receiving_blood_transfusion': 'History of Blood Transfusion',
        'receiving_unsterile_injections': 'History of Unsterile Injections',
        'prominent_veins_on_calf': 'Prominent Veins on Calf',
        'pus_filled_pimples': 'Pus-Filled Pimples',
        'yellow_crust_ooze': 'Yellow Crust Ooze',
        'small_dents_in_nails': 'Small Dents in Nails',
        'red_sore_around_nose': 'Red Sore Around Nose',
        'silver_like_dusting': 'Silver-Like Dusting on Skin',
    }
    if key in overrides:
        return overrides[key]
    cleaned = key.replace('_', ' ').replace('.', ' ').strip()
    return ' '.join(w.capitalize() for w in cleaned.split())

def parse_list_column(val):
    """Safely parse string representations of Python lists from CSV without using eval."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            res = ast.literal_eval(s)
            if isinstance(res, (list, tuple)):
                return [str(x).strip() for x in res if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [x.strip() for x in s.split(',') if x.strip()]

# Pre-build normalized lookup dictionaries on startup
description_map = {
    clean_disease_key(r['Disease']): str(r['Description']).strip()
    for _, r in description.iterrows()
}

precautions_map = {}
for _, r in precautions.iterrows():
    k = clean_disease_key(r['Disease'])
    items = []
    for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
        val = r.get(col)
        if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
            item_str = str(val).strip()
            item_str = item_str[0].upper() + item_str[1:] if item_str else item_str
            items.append(item_str)
    precautions_map[k] = items

medications_map = {
    clean_disease_key(r['Disease']): parse_list_column(r['Medication'])
    for _, r in medications.iterrows()
}

diets_map = {
    clean_disease_key(r['Disease']): parse_list_column(r['Diet'])
    for _, r in diets.iterrows()
}

workouts_map = {}
for _, r in workout.iterrows():
    k = clean_disease_key(r['disease'])
    val = r.get('workout')
    if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
        w_str = str(val).strip()
        w_str = w_str[0].upper() + w_str[1:] if w_str else w_str
        workouts_map.setdefault(k, []).append(w_str)

# Pre-generate catalogs for UI components
SYMPTOMS_CATALOG = [
    {'id': k, 'label': format_symptom_label(k)}
    for k in symptoms_dict.keys()
]

CONDITIONS_CATALOG = []
for idx, raw_d in diseases_list.items():
    ck = clean_disease_key(raw_d)
    display_title = display_disease_name(raw_d)
    desc = description_map.get(ck, "Clinical description currently being updated.")
    precs = precautions_map.get(ck, [])
    meds = medications_map.get(ck, [])
    diet_items = diets_map.get(ck, [])
    work_items = workouts_map.get(ck, [])

    CONDITIONS_CATALOG.append({
        'id': idx,
        'raw_name': raw_d,
        'name': display_title,
        'clean_key': ck,
        'description': desc,
        'precautions': precs,
        'medications': meds,
        'diets': diet_items,
        'workouts': work_items,
        'first_letter': display_title[0].upper() if display_title else '#'
    })

# Sort conditions alphabetically for easy browsing
CONDITIONS_CATALOG.sort(key=lambda x: x['name'].upper())

def helper(dis):
    """Retrieve supporting clinical data for a given disease."""
    ck = clean_disease_key(dis)
    desc = description_map.get(ck, "Clinical description currently being updated.")
    pre = precautions_map.get(ck, [])
    med = medications_map.get(ck, [])
    die = diets_map.get(ck, [])
    wrkout = workouts_map.get(ck, [])
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    """Predict disease using the trained SVC model."""
    input_vector = np.zeros(len(symptoms_dict))
    valid_count = 0

    for symptom in patient_symptoms:
        symptom = symptom.lower().strip()
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
            valid_count += 1

    # Guard: Do not run prediction if no valid symptoms were provided
    if valid_count == 0 or np.sum(input_vector) == 0:
        return "No Valid Symptoms"

    try:
        predicted_index = svc.predict([input_vector])[0]
        return diseases_list.get(predicted_index, "Unknown Disease")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error"

# =========================================================
# APPLICATION ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        symptoms_catalog=SYMPTOMS_CATALOG,
        conditions_catalog=CONDITIONS_CATALOG
    )

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return redirect(url_for('index'))

    symptoms_raw = request.form.get('symptoms', '').strip()
    message = None
    message_type = 'info'
    predicted_disease = None
    predicted_display = None
    dis_des = ""
    my_precautions = []
    medications_list = []
    my_diet = []
    workout_list = []
    submitted_symptoms = []

    # Validation: Check if input is provided
    if not symptoms_raw or symptoms_raw.lower() == "symptoms":
        message = "Please select at least one symptom from the search list to analyze."
        message_type = "error"
        return render_template(
            'index.html',
            message=message,
            message_type=message_type,
            symptoms_catalog=SYMPTOMS_CATALOG,
            conditions_catalog=CONDITIONS_CATALOG
        )

    # Split by comma and strip
    user_symptoms = [s.strip().lower() for s in symptoms_raw.split(',') if s.strip()]

    # Filter recognized canonical keys
    valid_keys = [s for s in user_symptoms if s in symptoms_dict]

    if not valid_keys:
        message = "None of the selected symptoms matched our medical database. Please pick from the autocomplete suggestions."
        message_type = "error"
        return render_template(
            'index.html',
            message=message,
            message_type=message_type,
            symptoms_catalog=SYMPTOMS_CATALOG,
            conditions_catalog=CONDITIONS_CATALOG,
            selected_symptoms_string=symptoms_raw
        )

    # Predict using the trained model
    predicted_disease = get_predicted_value(valid_keys)

    if predicted_disease in ["Prediction Error", "No Valid Symptoms", "Unknown Disease"]:
        message = "Unable to analyze symptoms. Please select at least one recognized symptom and try again."
        message_type = "error"
        return render_template(
            'index.html',
            message=message,
            message_type=message_type,
            symptoms_catalog=SYMPTOMS_CATALOG,
            conditions_catalog=CONDITIONS_CATALOG,
            selected_symptoms_string=','.join(valid_keys)
        )

    # Fetch supporting details
    dis_des, my_precautions, medications_list, my_diet, workout_list = helper(predicted_disease)
    predicted_display = display_disease_name(predicted_disease)

    # Build formatted list of submitted symptoms for display
    submitted_symptoms = [
        {'id': k, 'label': format_symptom_label(k)}
        for k in valid_keys
    ]

    return render_template(
        'index.html',
        predicted_disease=predicted_disease,
        predicted_disease_display=predicted_display,
        dis_des=dis_des,
        my_precautions=my_precautions,
        medications=medications_list,
        my_diet=my_diet,
        workout=workout_list,
        submitted_symptoms=submitted_symptoms,
        selected_symptoms_string=','.join(valid_keys),
        message="Symptom analysis completed successfully.",
        message_type="success",
        symptoms_catalog=SYMPTOMS_CATALOG,
        conditions_catalog=CONDITIONS_CATALOG
    )

@app.route('/diseases.html')
def diseases():
    """Repair missing route: Provide dedicated Condition Explorer page."""
    return render_template(
        "diseases.html",
        conditions_catalog=CONDITIONS_CATALOG,
        symptoms_catalog=SYMPTOMS_CATALOG
    )

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route('/signin.html')
def signin():
    return render_template("signin.html")

@app.route('/api/symptoms')
def api_symptoms():
    """JSON API to retrieve full catalog of recognized symptoms."""
    return jsonify(SYMPTOMS_CATALOG)

@app.route('/api/conditions')
def api_conditions():
    """JSON API to retrieve catalog of conditions."""
    return jsonify(CONDITIONS_CATALOG)

if __name__ == '__main__':
    app.run(debug=True)
