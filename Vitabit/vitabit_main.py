import base64
import json
import os
import re
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps

from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set. Copy .env.example to .env and add your MongoDB URI.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "vitabit-dev-secret")

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client["Vitabit-Database"]
supplements_collection = db["supplements"]
users_collection = db["users"]
profiles_collection = db["profiles"]
tracked_items_collection = db["tracked_items"]
completion_logs_collection = db["completion_logs"]
meal_analyses_collection = db["meal_analyses"]

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

NUTRIENT_ALIASES = {
    "Vitamin D": ["vitamin d", "vitamin d3", "d3"],
    "Vitamin C": ["vitamin c", "ascorbic acid"],
    "Vitamin B12": ["vitamin b12", "b12", "cobalamin"],
    "Folate": ["folate", "folic acid", "b9", "methylfolate"],
    "Calcium": ["calcium"],
    "Iron": ["iron"],
    "Magnesium": ["magnesium"],
    "Zinc": ["zinc"],
}

NUTRIENT_TARGET_NAMES = {
    "Vitamin D": "Vitamin D3",
    "Vitamin C": "Vitamin C",
    "Vitamin B12": "Vitamin B12",
    "Folate": "Vitamin B9",
    "Calcium": "Calcium",
    "Iron": "Iron",
    "Magnesium": "Magnesium",
    "Zinc": "Zinc",
}

DEFAULT_TARGET_REFERENCES = {
}


def utcnow():
    return datetime.now(timezone.utc)


def today_key():
    return datetime.now().strftime("%Y-%m-%d")


def is_onboarding_complete(user):
    return bool(user.get("onboarding_completed"))


def serialize_document(document):
    if not document:
        return None
    serialized = dict(document)
    serialized["_id"] = str(serialized["_id"])
    if "user_id" in serialized and isinstance(serialized["user_id"], ObjectId):
        serialized["user_id"] = str(serialized["user_id"])
    if "profile_id" in serialized and isinstance(serialized["profile_id"], ObjectId):
        serialized["profile_id"] = str(serialized["profile_id"])
    if "tracked_item_id" in serialized and isinstance(serialized["tracked_item_id"], ObjectId):
        serialized["tracked_item_id"] = str(serialized["tracked_item_id"])
    return serialized


def get_openai_client():
    if not OPENAI_API_KEY:
        return None, "OPENAI_API_KEY is not configured."
    if OpenAI is None:
        return None, "OpenAI package is not installed. Run pip install -r requirements.txt."
    return OpenAI(api_key=OPENAI_API_KEY), None


def get_response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    fragments = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                fragments.append(text)
    return "\n".join(fragments).strip()


def load_catalog(limit=None):
    cursor = supplements_collection.find({}, {"_id": 0}).sort("name", 1)
    if limit is not None:
        cursor = cursor.limit(limit)
    return list(cursor)


def build_catalog_context(max_items=20):
    items = load_catalog(limit=max_items)
    return "\n".join(
        f"- {item.get('name')} [{item.get('category', 'Supplement')}]: {item.get('benefit', 'General wellness support')}"
        for item in items
    )


def extract_json_block(text):
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def find_nutrients_in_text(*values):
    text = " ".join(normalize_text(value) for value in values if value)
    matches = set()
    if not text:
        return matches

    for nutrient, aliases in NUTRIENT_ALIASES.items():
        if any(alias in text for alias in aliases):
            matches.add(nutrient)
    return matches


def build_target_reference_map():
    refs = {}
    for nutrient, catalog_name in NUTRIENT_TARGET_NAMES.items():
        if catalog_name:
            item = supplements_collection.find_one({"name": catalog_name})
            if item:
                refs[nutrient] = {
                    "male": item.get("male_daily_intake") or "",
                    "female": item.get("female_daily_intake") or "",
                    "general": item.get("intake_note") or "",
                }
                continue
        refs[nutrient] = DEFAULT_TARGET_REFERENCES.get(
            nutrient,
            {"male": "", "female": "", "general": "General wellness reference"},
        )
    return refs


def get_target_reference(nutrient, profile, target_map):
    sex = normalize_text(profile.get("sex", ""))
    ref = target_map.get(nutrient, {})
    if sex == "male" and ref.get("male"):
        return ref["male"]
    if sex == "female" and ref.get("female"):
        return ref["female"]
    return ref.get("general") or ref.get("female") or ref.get("male") or "General wellness reference"


def parse_date_key(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def classify_coverage(avg_score):
    if avg_score >= 2.4:
        return "Strong"
    if avg_score >= 1.2:
        return "Okay"
    return "Low"


def build_dashboard_window(user, current_profile, tracked_items, days, target_map):
    if not current_profile:
        return {
            "meal_count": 0,
            "active_stack": {"total": 0, "supplements": 0, "medications": 0},
            "adherence": {"completed": 0, "expected": 0, "rate": 0},
            "coverage": [],
            "often_missing": [],
            "often_covered": [],
            "recent_summaries": [],
        }

    today = datetime.now().date()
    start_date = today - timedelta(days=days - 1)
    meal_docs = list(
        meal_analyses_collection.find(
            {
                "user_id": user["_id"],
                "profile_id": current_profile["_id"],
                "created_at": {"$gte": datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)},
            }
        ).sort("created_at", -1)
    )
    completion_docs = list(
        completion_logs_collection.find(
            {
                "user_id": user["_id"],
                "profile_id": current_profile["_id"],
                "completed": True,
            }
        )
    )

    nutrient_scores = {name: 0.0 for name in NUTRIENT_ALIASES}
    gap_counts = {name: 0 for name in NUTRIENT_ALIASES}
    covered_counts = {name: 0 for name in NUTRIENT_ALIASES}

    for meal in meal_docs:
        for nutrient in find_nutrients_in_text(*(meal.get("estimated_nutrients") or [])):
            nutrient_scores[nutrient] += 1.0
        for nutrient in find_nutrients_in_text(*(meal.get("likely_covered") or [])):
            nutrient_scores[nutrient] += 1.6
            covered_counts[nutrient] += 1
        for nutrient in find_nutrients_in_text(*(meal.get("likely_gaps") or [])):
            gap_counts[nutrient] += 1

    tracked_lookup = {str(item["_id"]): item for item in tracked_items}
    completed_count = 0
    for log in completion_docs:
        log_date = parse_date_key(log.get("date_key", ""))
        if not log_date or log_date < start_date:
            continue
        completed_count += 1
        tracked_item = tracked_lookup.get(str(log.get("tracked_item_id")))
        if not tracked_item:
            continue
        nutrients = find_nutrients_in_text(tracked_item.get("name"), tracked_item.get("category"), tracked_item.get("notes"))
        for nutrient in nutrients:
            nutrient_scores[nutrient] += 2.6
            covered_counts[nutrient] += 1

    custom_coverage = []
    for item in tracked_items:
        if not item.get("show_in_dashboard"):
            continue
        item_logs = [
            log for log in completion_docs
            if str(log.get("tracked_item_id")) == item["_id"] and parse_date_key(log.get("date_key", "")) and parse_date_key(log.get("date_key", "")) >= start_date
        ]
        completed_item_count = len(item_logs)
        expected_item_count = days
        custom_coverage.append(
            {
                "name": item.get("name", "Tracked Item"),
                "percent": min(100, round((completed_item_count / expected_item_count) * 100)) if expected_item_count else 0,
                "status": classify_coverage((completed_item_count / expected_item_count) * 2.6 if expected_item_count else 0),
                "target_reference": "Custom dashboard item from your tracked stack",
            }
        )

    active_stack = {
        "total": len(tracked_items),
        "supplements": len([item for item in tracked_items if item.get("item_type") == "supplement"]),
        "medications": len([item for item in tracked_items if item.get("item_type") == "medication"]),
    }
    expected_count = active_stack["total"] * days
    adherence_rate = round((completed_count / expected_count) * 100) if expected_count else 0

    coverage = []
    for nutrient, score in nutrient_scores.items():
        avg_score = score / days
        percent = min(100, round((avg_score / 2.6) * 100))
        status = classify_coverage(avg_score)
        coverage.append(
            {
                "name": nutrient,
                "score": round(avg_score, 2),
                "percent": percent,
                "status": status,
                "target_reference": get_target_reference(nutrient, current_profile, target_map),
            }
        )

    coverage.sort(key=lambda item: (-item["percent"], item["name"]))
    often_missing = [item["name"] for item in sorted(coverage, key=lambda item: (item["percent"], item["name"]))[:4]]
    often_covered = [item["name"] for item in coverage[:4] if item["percent"] >= 45]

    return {
        "meal_count": len(meal_docs),
        "active_stack": active_stack,
        "adherence": {"completed": completed_count, "expected": expected_count, "rate": adherence_rate},
        "coverage": coverage,
        "custom_coverage": custom_coverage,
        "often_missing": often_missing,
        "often_covered": often_covered,
        "recent_summaries": [meal.get("meal_summary", "") for meal in meal_docs[:3] if meal.get("meal_summary")],
    }


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        return users_collection.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


def get_current_profile(user_id=None):
    profile_id = session.get("current_profile_id")
    if not profile_id:
        return None

    query = {"_id": ObjectId(profile_id)}
    if user_id:
        query["user_id"] = ObjectId(user_id)

    try:
        return profiles_collection.find_one(query)
    except Exception:
        return None


def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must include at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least one special character."
    return None


def send_welcome_email(email):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM_EMAIL")

    if not all([smtp_host, smtp_port, smtp_username, smtp_password, smtp_from]):
        return False

    message = EmailMessage()
    message["Subject"] = "Welcome to Vitabit"
    message["From"] = smtp_from
    message["To"] = email
    message.set_content(
        "Welcome to Vitabit.\n\n"
        "Your account is ready. You can track supplements, medications, meal analyses, and household wellness routines.\n\n"
        "For diagnosis, treatment, or medication changes, consult a licensed clinician or pharmacist."
    )

    try:
        with smtplib.SMTP(smtp_host, int(smtp_port)) as smtp:
            smtp.starttls()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(message)
        return True
    except Exception as exc:
        print(f"Welcome email failed: {exc}")
        return False


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required."}), 401
        return fn(user, *args, **kwargs)

    return wrapped


def ensure_default_profile(user_id, email):
    profile = profiles_collection.find_one({"user_id": ObjectId(user_id), "is_primary": True})
    if profile:
        return profile

    default_profile = {
        "user_id": ObjectId(user_id),
        "name": email.split("@")[0] or "Me",
        "relationship": "Self",
        "age_group": "Adult",
        "sex": "",
        "age": "",
        "height_cm": "",
        "weight_kg": "",
        "wellness_goal": "",
        "diet_style": "",
        "is_primary": True,
        "created_at": utcnow(),
    }
    inserted = profiles_collection.insert_one(default_profile)
    return profiles_collection.find_one({"_id": inserted.inserted_id})


def get_household_payload(user):
    user_id = user["_id"]
    profiles = [serialize_document(profile) for profile in profiles_collection.find({"user_id": user_id}).sort("created_at", 1)]
    current_profile = get_current_profile(user_id=str(user_id))
    if not current_profile and profiles:
        current_profile = profiles_collection.find_one({"_id": ObjectId(profiles[0]["_id"])})
        session["current_profile_id"] = str(current_profile["_id"])

    current_profile_serialized = serialize_document(current_profile)
    tracked_items = []
    completion_map = {}
    meal_history = []
    if current_profile:
        cursor = tracked_items_collection.find(
            {"user_id": user_id, "profile_id": current_profile["_id"]}
        ).sort("schedule_time", 1)
        tracked_items = [serialize_document(item) for item in cursor]
        log_cursor = completion_logs_collection.find(
            {
                "user_id": user_id,
                "profile_id": current_profile["_id"],
                "date_key": today_key(),
            }
        )
        completion_map = {
            str(log["tracked_item_id"]): bool(log.get("completed", False))
            for log in log_cursor
        }
        meal_cursor = meal_analyses_collection.find(
            {"user_id": user_id, "profile_id": current_profile["_id"]}
        ).sort("created_at", -1).limit(5)
        meal_history = [serialize_document(item) for item in meal_cursor]

    return {
        "user": {"email": user["email"], "_id": str(user_id)},
        "onboarding_completed": is_onboarding_complete(user),
        "profiles": profiles,
        "current_profile": current_profile_serialized,
        "tracked_items": tracked_items,
        "completion_map": completion_map,
        "meal_history": meal_history,
        "today_key": today_key(),
        "max_profiles": 4,
    }


def build_dashboard_summary(user, current_profile):
    if not current_profile:
        return {
            "default_range": "today",
            "windows": {
                "today": build_dashboard_window(user, current_profile, [], 1, {}),
                "7d": build_dashboard_window(user, current_profile, [], 7, {}),
                "30d": build_dashboard_window(user, current_profile, [], 30, {}),
            },
        }

    tracked_items = [serialize_document(item) for item in tracked_items_collection.find(
        {"user_id": user["_id"], "profile_id": current_profile["_id"]}
    )]
    target_map = build_target_reference_map()
    return {
        "default_range": "today",
        "windows": {
            "today": build_dashboard_window(user, current_profile, tracked_items, 1, target_map),
            "7d": build_dashboard_window(user, current_profile, tracked_items, 7, target_map),
            "30d": build_dashboard_window(user, current_profile, tracked_items, 30, target_map),
        },
    }


@app.route("/")
def index():
    user = get_current_user()
    if user and not is_onboarding_complete(user):
        return redirect(url_for("onboarding_page"))
    return render_template("index.html")


@app.route("/profile")
def profile_page():
    user = get_current_user()
    if not user:
        return redirect(url_for("auth_page"))
    if not is_onboarding_complete(user):
        return redirect(url_for("onboarding_page"))
    return render_template("profile.html")


@app.route("/auth")
def auth_page():
    if get_current_user():
        return redirect(url_for("index"))
    return render_template("auth.html", error=request.args.get("error", ""), mode=request.args.get("mode", "login"))


@app.get("/onboarding")
def onboarding_page():
    user = get_current_user()
    if not user:
        return redirect(url_for("auth_page"))
    if is_onboarding_complete(user):
        return redirect(url_for("index"))

    profile = ensure_default_profile(user["_id"], user["email"])
    return render_template("onboarding.html", email=user["email"], profile=serialize_document(profile))


@app.post("/onboarding")
def complete_onboarding():
    user = get_current_user()
    if not user:
        return redirect(url_for("auth_page"))

    skip = request.form.get("skip") == "1"
    profile = ensure_default_profile(user["_id"], user["email"])

    if not skip:
        sex = (request.form.get("sex") or "").strip()
        age = (request.form.get("age") or "").strip()
        height_cm = (request.form.get("height_cm") or "").strip()
        weight_kg = (request.form.get("weight_kg") or "").strip()

        profiles_collection.update_one(
            {"_id": profile["_id"], "user_id": user["_id"]},
            {
                "$set": {
                    "sex": sex,
                    "age": age,
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "updated_at": utcnow(),
                }
            },
        )

    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"onboarding_completed": True, "updated_at": utcnow()}},
    )
    return redirect(url_for("index"))


@app.post("/auth/register")
def register():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    confirm_password = request.form.get("confirm_password") or ""

    if not email or not password:
        return redirect(url_for("auth_page", mode="register", error="Email and password are required."))
    if password != confirm_password:
        return redirect(url_for("auth_page", mode="register", error="Passwords do not match."))
    password_error = validate_password(password)
    if password_error:
        return redirect(url_for("auth_page", mode="register", error=password_error))
    if users_collection.find_one({"email": email}):
        return redirect(url_for("auth_page", mode="register", error="An account already exists for that email."))

    new_user = {
        "email": email,
        "password_hash": generate_password_hash(password),
        "onboarding_completed": False,
        "created_at": utcnow(),
    }
    inserted = users_collection.insert_one(new_user)
    profile = ensure_default_profile(inserted.inserted_id, email)
    send_welcome_email(email)

    session["user_id"] = str(inserted.inserted_id)
    session["current_profile_id"] = str(profile["_id"])
    return redirect(url_for("onboarding_page"))


@app.post("/auth/login")
def login():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password_hash"], password):
        return redirect(url_for("auth_page", mode="login", error="Invalid email or password."))

    profile = ensure_default_profile(user["_id"], user["email"])
    session["user_id"] = str(user["_id"])
    session["current_profile_id"] = str(profile["_id"])
    if not is_onboarding_complete(user):
        return redirect(url_for("onboarding_page"))
    return redirect(url_for("index"))


@app.post("/auth/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.get("/api/me")
def me():
    user = get_current_user()
    if not user:
        return jsonify({"authenticated": False})

    payload = get_household_payload(user)
    payload["dashboard_summary"] = build_dashboard_summary(
        user, get_current_profile(user_id=str(user["_id"]))
    )
    payload["authenticated"] = True
    return jsonify(payload)


@app.get("/catalog")
def get_catalog():
    try:
        result = []
        for item in supplements_collection.find().sort("name", 1):
            result.append(serialize_document(item))
    except PyMongoError as exc:
        print(f"MongoDB catalog query failed: {exc}")
        return jsonify({"error": "Database query failed"}), 503

    return jsonify(result)


@app.route("/search")
@app.route("/search/<keyword>")
def search_vitamin_info(keyword=None):
    result = []
    keyword = (keyword or request.args.get("keyword", "")).strip()
    if not keyword:
        return jsonify([])

    escaped_keyword = re.escape(keyword)
    keyword_length = len(keyword)

    if keyword_length == 1:
        # Single-character searches should feel precise, not like a full-text scan.
        query = {"name": {"$regex": f"^{escaped_keyword}", "$options": "i"}}
    elif keyword_length == 2:
        # Short searches can widen slightly, but still prefer prefix-style matches.
        query = {
            "$or": [
                {"name": {"$regex": f"^{escaped_keyword}", "$options": "i"}},
                {"keywords": {"$regex": f"^{escaped_keyword}", "$options": "i"}},
            ]
        }
    else:
        query = {
            "$or": [
                {"name": {"$regex": escaped_keyword, "$options": "i"}},
                {"category": {"$regex": escaped_keyword, "$options": "i"}},
                {"benefit": {"$regex": escaped_keyword, "$options": "i"}},
                {"keywords": {"$regex": escaped_keyword, "$options": "i"}},
            ]
        }

    try:
        for item in supplements_collection.find(query).sort("name", 1):
            result.append(serialize_document(item))
    except PyMongoError as exc:
        print(f"MongoDB query failed: {exc}")
        return jsonify({"error": "Database query failed"}), 503

    print(f"Search query: '{keyword}' | Found: {len(result)} items")
    return jsonify(result)


@app.post("/api/profiles")
@login_required
def create_profile(user):
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    relationship = (payload.get("relationship") or "").strip()
    sex = (payload.get("sex") or "").strip()
    age = (payload.get("age") or "").strip()
    height_cm = (payload.get("height_cm") or "").strip()
    height_value = (payload.get("height_value") or "").strip()
    height_unit = (payload.get("height_unit") or "").strip()
    weight_kg = (payload.get("weight_kg") or "").strip()
    weight_value = (payload.get("weight_value") or "").strip()
    weight_unit = (payload.get("weight_unit") or "").strip()
    wellness_goal = (payload.get("wellness_goal") or "").strip()
    diet_style = (payload.get("diet_style") or "").strip()

    if not name:
        return jsonify({"error": "Profile name is required."}), 400

    profile_count = profiles_collection.count_documents({"user_id": user["_id"]})
    if profile_count >= 4:
        return jsonify({"error": "You can store up to 4 household profiles in this MVP."}), 400

    profile = {
        "user_id": user["_id"],
        "name": name,
        "relationship": relationship or "Household",
        "sex": sex,
        "age": age,
        "height_cm": height_cm,
        "height_value": height_value,
        "height_unit": height_unit,
        "weight_kg": weight_kg,
        "weight_value": weight_value,
        "weight_unit": weight_unit,
        "wellness_goal": wellness_goal,
        "diet_style": diet_style,
        "is_primary": False,
        "created_at": utcnow(),
    }
    inserted = profiles_collection.insert_one(profile)
    session["current_profile_id"] = str(inserted.inserted_id)
    return jsonify({"ok": True, "profile": serialize_document(profiles_collection.find_one({"_id": inserted.inserted_id}))})


@app.post("/api/profiles/<profile_id>/select")
@login_required
def select_profile(user, profile_id):
    profile = profiles_collection.find_one({"_id": ObjectId(profile_id), "user_id": user["_id"]})
    if not profile:
        return jsonify({"error": "Profile not found."}), 404

    session["current_profile_id"] = str(profile["_id"])
    return jsonify({"ok": True, "profile": serialize_document(profile)})


@app.patch("/api/profiles/<profile_id>")
@login_required
def update_profile(user, profile_id):
    profile = profiles_collection.find_one({"_id": ObjectId(profile_id), "user_id": user["_id"]})
    if not profile:
        return jsonify({"error": "Profile not found."}), 404

    payload = request.get_json(silent=True) or {}
    allowed_fields = [
        "name",
        "relationship",
        "sex",
        "age",
        "height_cm",
        "height_value",
        "height_unit",
        "weight_kg",
        "weight_value",
        "weight_unit",
        "wellness_goal",
        "diet_style",
    ]
    updates = {}
    for field in allowed_fields:
        if field in payload:
            updates[field] = (payload.get(field) or "").strip()

    if not updates:
        return jsonify({"error": "No profile updates provided."}), 400

    profiles_collection.update_one(
        {"_id": profile["_id"], "user_id": user["_id"]},
        {"$set": {**updates, "updated_at": utcnow()}},
    )
    updated = profiles_collection.find_one({"_id": profile["_id"]})
    return jsonify({"ok": True, "profile": serialize_document(updated)})


@app.delete("/api/profiles/<profile_id>")
@login_required
def delete_profile(user, profile_id):
    profile = profiles_collection.find_one({"_id": ObjectId(profile_id), "user_id": user["_id"]})
    if not profile:
        return jsonify({"error": "Profile not found."}), 404
    if profile.get("is_primary"):
        return jsonify({"error": "Primary profile cannot be deleted in this MVP."}), 400

    profiles_collection.delete_one({"_id": profile["_id"], "user_id": user["_id"]})
    tracked_items_collection.delete_many({"user_id": user["_id"], "profile_id": profile["_id"]})
    completion_logs_collection.delete_many({"user_id": user["_id"], "profile_id": profile["_id"]})
    meal_analyses_collection.delete_many({"user_id": user["_id"], "profile_id": profile["_id"]})

    current_profile = get_current_profile(user_id=str(user["_id"]))
    if current_profile and str(current_profile["_id"]) == profile_id:
        fallback = profiles_collection.find_one({"user_id": user["_id"]}, sort=[("created_at", 1)])
        if fallback:
            session["current_profile_id"] = str(fallback["_id"])

    return jsonify({"ok": True})


@app.get("/api/tracked-items")
@login_required
def get_tracked_items(user):
    current_profile = get_current_profile(user_id=str(user["_id"]))
    if not current_profile:
        return jsonify([])

    cursor = tracked_items_collection.find(
        {"user_id": user["_id"], "profile_id": current_profile["_id"]}
    ).sort("schedule_time", 1)
    return jsonify([serialize_document(item) for item in cursor])


@app.post("/api/tracked-items")
@login_required
def create_tracked_item(user):
    current_profile = get_current_profile(user_id=str(user["_id"]))
    if not current_profile:
        return jsonify({"error": "Select a profile first."}), 400

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    item_type = (payload.get("item_type") or "supplement").strip().lower()
    category = (payload.get("category") or "").strip()
    schedule_time = (payload.get("schedule_time") or "").strip()
    notes = (payload.get("notes") or "").strip()
    source_type = (payload.get("source_type") or "manual").strip().lower()

    if not name:
        return jsonify({"error": "Item name is required."}), 400
    if item_type not in {"supplement", "medication"}:
        return jsonify({"error": "Item type must be supplement or medication."}), 400

    tracked_item = {
        "user_id": user["_id"],
        "profile_id": current_profile["_id"],
        "name": name,
        "item_type": item_type,
        "category": category,
        "schedule_time": schedule_time or "08:00",
        "notes": notes,
        "source_type": source_type,
        "show_in_dashboard": bool(payload.get("show_in_dashboard")),
        "created_at": utcnow(),
    }
    inserted = tracked_items_collection.insert_one(tracked_item)
    return jsonify({"ok": True, "item": serialize_document(tracked_items_collection.find_one({"_id": inserted.inserted_id}))})


@app.post("/api/tracked-items/<item_id>/dashboard")
@login_required
def set_tracked_item_dashboard(user, item_id):
    current_profile = get_current_profile(user_id=str(user["_id"]))
    if not current_profile:
        return jsonify({"error": "Select a profile first."}), 400

    payload = request.get_json(silent=True) or {}
    show_in_dashboard = bool(payload.get("show_in_dashboard"))
    result = tracked_items_collection.update_one(
        {"_id": ObjectId(item_id), "user_id": user["_id"], "profile_id": current_profile["_id"]},
        {"$set": {"show_in_dashboard": show_in_dashboard, "updated_at": utcnow()}},
    )
    if not result.matched_count:
        return jsonify({"error": "Tracked item not found."}), 404
    return jsonify({"ok": True, "show_in_dashboard": show_in_dashboard})


@app.post("/api/tracked-items/<item_id>/completion")
@login_required
def set_tracked_item_completion(user, item_id):
    current_profile = get_current_profile(user_id=str(user["_id"]))
    if not current_profile:
        return jsonify({"error": "Select a profile first."}), 400

    payload = request.get_json(silent=True) or {}
    completed = bool(payload.get("completed"))
    tracked_item = tracked_items_collection.find_one(
        {"_id": ObjectId(item_id), "user_id": user["_id"], "profile_id": current_profile["_id"]}
    )
    if not tracked_item:
        return jsonify({"error": "Tracked item not found."}), 404

    completion_logs_collection.update_one(
        {
            "user_id": user["_id"],
            "profile_id": current_profile["_id"],
            "tracked_item_id": tracked_item["_id"],
            "date_key": today_key(),
        },
        {
            "$set": {
                "completed": completed,
                "updated_at": utcnow(),
            },
            "$setOnInsert": {
                "created_at": utcnow(),
            },
        },
        upsert=True,
    )
    return jsonify({"ok": True, "completed": completed})


@app.delete("/api/tracked-items/<item_id>")
@login_required
def delete_tracked_item(user, item_id):
    result = tracked_items_collection.delete_one({"_id": ObjectId(item_id), "user_id": user["_id"]})
    if not result.deleted_count:
        return jsonify({"error": "Tracked item not found."}), 404
    return jsonify({"ok": True})


@app.post("/assistant/chat")
def assistant_chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400

    openai_client, error_message = get_openai_client()
    if error_message:
        return jsonify({"error": error_message}), 503

    user = get_current_user()
    household_context = ""
    if user:
        household = get_household_payload(user)
        current_profile = household.get("current_profile") or {}
        tracked_items = household.get("tracked_items") or []
        meal_history = household.get("meal_history") or []
        household_context = (
            f"Current profile: {current_profile.get('name', 'Unknown')} | "
            f"Relationship: {current_profile.get('relationship', '')} | "
            f"Goal: {current_profile.get('wellness_goal', '')} | "
            f"Recent metrics: sex={current_profile.get('sex', '')}, age={current_profile.get('age', '')}, height_cm={current_profile.get('height_cm', '')}, weight_kg={current_profile.get('weight_kg', '')}\n"
            "Tracked items:\n"
            + "\n".join(
                f"- {item.get('name')} ({item.get('item_type')}) at {item.get('schedule_time', 'unscheduled')}"
                for item in tracked_items[:12]
            )
            + "\nRecent meal history:\n"
            + "\n".join(
                f"- {meal.get('meal_summary', 'Meal')} | gaps: {', '.join(meal.get('likely_gaps', []) or ['none noted'])}"
                for meal in meal_history[:5]
            )
        )

    system_prompt = (
        "You are Vitabit Coach, an AI assistant for a wellness tracking app that covers supplements, medications, meals, and routines. "
        "Answer in practical plain language. Do not diagnose disease. For treatment decisions, medication changes, or condition-specific care, advise the user to consult a licensed clinician or pharmacist. "
        "If the user asks about supplements, prefer items from this in-app catalog when relevant.\n\n"
        "Return valid JSON with this shape only:\n"
        "{"
        "\"answer\": string, "
        "\"suggested_supplements\": [string], "
        "\"reminder_tips\": [string], "
        "\"safety_note\": string"
        "}\n\n"
        f"Catalog snapshot:\n{build_catalog_context()}\n\n"
        f"User household context:\n{household_context}"
    )

    try:
        response = openai_client.responses.create(
            model=OPENAI_CHAT_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": message}]},
            ],
        )
    except Exception as exc:
        print(f"OpenAI chat request failed: {exc}")
        return jsonify({"error": "AI assistant request failed."}), 503

    raw_text = get_response_text(response)
    parsed = extract_json_block(raw_text) or {
        "answer": raw_text or "No response returned.",
        "suggested_supplements": [],
        "reminder_tips": [],
        "safety_note": "General guidance only. Check medications and health conditions with a clinician.",
    }
    return jsonify(parsed)


@app.post("/assistant/analyze-meal")
def analyze_meal():
    uploaded_file = request.files.get("image")
    notes = (request.form.get("notes") or "").strip()

    if not uploaded_file:
        return jsonify({"error": "Image is required."}), 400

    image_bytes = uploaded_file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded image is empty."}), 400

    mime_type = uploaded_file.mimetype or "image/jpeg"
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{base64_image}"

    openai_client, error_message = get_openai_client()
    if error_message:
        return jsonify({"error": error_message}), 503

    system_prompt = (
        "You analyze meal photos for a wellness tracking web app. "
        "Estimate likely nutrients from visible food, but explicitly frame the output as an estimate. "
        "Return valid JSON with this exact shape only:\n"
        "{"
        "\"meal_summary\": string, "
        "\"estimated_nutrients\": [string], "
        "\"likely_covered\": [string], "
        "\"likely_gaps\": [string], "
        "\"supplement_ideas\": [string], "
        "\"confidence_note\": string"
        "}"
    )

    user_text = (
        "Analyze this meal image for likely vitamin, mineral, protein, fiber, omega-3, and probiotic coverage. "
        "Keep the output practical for a household wellness dashboard."
    )
    if notes:
        user_text += f" User notes: {notes}"

    try:
        response = openai_client.responses.create(
            model=OPENAI_VISION_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },
            ],
        )
    except Exception as exc:
        print(f"OpenAI vision request failed: {exc}")
        return jsonify({"error": "Meal analysis request failed."}), 503

    raw_text = get_response_text(response)
    parsed = extract_json_block(raw_text) or {
        "meal_summary": "Meal analysis could not be structured.",
        "estimated_nutrients": [],
        "likely_covered": [],
        "likely_gaps": [],
        "supplement_ideas": [],
        "confidence_note": raw_text or "Estimate unavailable.",
    }

    user = get_current_user()
    if user:
        current_profile = get_current_profile(user_id=str(user["_id"]))
        if current_profile:
            meal_analyses_collection.insert_one(
                {
                    "user_id": user["_id"],
                    "profile_id": current_profile["_id"],
                    "notes": notes,
                    "meal_summary": parsed.get("meal_summary", ""),
                    "estimated_nutrients": parsed.get("estimated_nutrients", []),
                    "likely_covered": parsed.get("likely_covered", []),
                    "likely_gaps": parsed.get("likely_gaps", []),
                    "supplement_ideas": parsed.get("supplement_ideas", []),
                    "confidence_note": parsed.get("confidence_note", ""),
                    "created_at": utcnow(),
                }
            )

    return jsonify(parsed)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
