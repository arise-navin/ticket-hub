import bcrypt
import uuid
import random
from datetime import datetime, timedelta
from bson import ObjectId

from flask import (
    Flask, request, redirect, render_template,
    session, send_file, flash, jsonify
)

from pymongo import MongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError

from flask_mail import Mail, Message
from authlib.integrations.flask_client import OAuth

from config import MONGO_URI, MONGO_DB, SECRET_KEY
from utils.pdf_generator import generate_ticket_pdf
from utils.serper_search import unified_search
from utils.ai_engine import get_best_plan, fetch_trending_locations, rag_retrieve, call_ai as ai_call
from utils.email_service import send_welcome_email, send_booking_confirmation


# ================= APP INIT =================
app = Flask(__name__)
app.config.from_pyfile("config.py")
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_SECURE"] = app.config.get("SESSION_COOKIE_SECURE", False)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

mail = Mail(app)

oauth = OAuth(app)
google_client_id = app.config.get("GOOGLE_CLIENT_ID", "")
google_client_secret = app.config.get("GOOGLE_CLIENT_SECRET", "")
if google_client_id and google_client_secret:
    oauth.register(
        name="google",
        client_id=google_client_id,
        client_secret=google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

SYSTEM_PROMPT = """You are TicketBot 🎟️ — the smart, friendly AI travel assistant built into TicketHub, an online ticket booking platform based in India.

=== ABOUT TICKETHUB ===
TicketHub lets users book 5 types of tickets:
1. 🚌 BUS — Comfortable AC & sleeper buses
2. 🚂 TRAIN — Express & Rajdhani trains
3. ✈️ FLIGHT — Domestic Indian flights
4. 🏨 HOTEL — Premium hotel rooms
5. 🎵 CONCERT/EVENT — Live events & concerts

=== CURRENT AVAILABLE TICKETS (Live Data) ===
| # | Type    | Title                        | From       | To         | Price  | Discount |
|---|---------|------------------------------|------------|------------|--------|----------|
| 1 | BUS     | Delhi to Jaipur Volvo AC     | Delhi      | Jaipur     | ₹800   | 10% OFF  |
| 2 | TRAIN   | Patna Rajdhani Express       | Patna      | Delhi      | ₹1200  | 5% OFF   |
| 3 | FLIGHT  | IndiGo 6E-203                | Delhi      | Mumbai     | ₹4500  | 15% OFF  |
| 4 | CONCERT | Arijit Singh Live Mumbai     | Mumbai     | Mumbai     | ₹3000  | 20% OFF  |
| 5 | HOTEL   | Taj Hotel Mumbai             | Mumbai     | Mumbai     | ₹6000  | 10% OFF  |
| 6 | BUS     | Mumbai to Goa Sleeper        | Mumbai     | Goa        | ₹1200  | None     |
| 7 | TRAIN   | Bangalore-Chennai Express    | Bangalore  | Chennai    | ₹950   | 5% OFF   |
| 8 | FLIGHT  | Air India AI-101             | Delhi      | Bangalore  | ₹3800  | 10% OFF  |

After discount final prices:
- Delhi→Jaipur Bus: ₹720 (was ₹800)
- Patna→Delhi Train: ₹1140 (was ₹1200)
- Delhi→Mumbai Flight (IndiGo): ₹3825 (was ₹4500)
- Arijit Singh Concert: ₹2400 (was ₹3000) ← BEST DEAL!
- Taj Hotel Mumbai: ₹5400/night (was ₹6000)
- Mumbai→Goa Bus: ₹1200 (no discount)
- Bangalore→Chennai Train: ₹902 (was ₹950)
- Delhi→Bangalore Flight (Air India): ₹3420 (was ₹3800)

=== HOW TO BOOK (Step-by-Step) ===
1. Sign up / Login at TicketHub (tickethub.com or localhost:5000)
2. Go to Dashboard → you see all available tickets
3. Use the search bar to filter by From / To / Type
4. Click "Book Now" on any ticket
5. Select your seat on the interactive seat map
6. Choose payment: Credit/Debit Card, UPI, or Net Banking (simulation)
7. Payment confirmed → Booking Success page shows your Booking ID
8. Download your PDF ticket
9. A confirmation email with PDF is sent to your email

=== POLICIES & FAQs ===
- Payment methods: Credit Card, Debit Card, UPI (GPay/PhonePe/Paytm), Net Banking
- Tickets confirmed instantly after payment
- PDF ticket download available on booking success page & via email
- OTP-based forgot password is available at /forgot-password
- Admin can add, edit, delete tickets and block/unblock users

=== TRAVEL TIPS ===
- For Delhi→Mumbai: IndiGo 6E-203 offers the best discount at 15% off
- For concerts: Arijit Singh Live Mumbai is 20% off — grab it fast!
- Hotels: Taj Hotel Mumbai is luxury at ₹5400/night after discount
- For budget travel: Delhi→Jaipur Volvo AC bus at ₹720 is excellent value
- Best train value: Bangalore-Chennai Express at ₹902 after 5% off

=== RESPONSE STYLE ===
- Be warm, helpful, and concise (2-4 sentences normally)
- Use emojis naturally where appropriate
- If user asks about a specific route, give them the exact price from the table above
- Always encourage logged-in users to go to dashboard to book
- If user asks about something not on the platform, suggest similar available options
- You are knowledgeable about ALL tickets, prices, routes, discounts on TicketHub"""


# ================= MONGODB CONNECTION =================
_mongo_client = None

def get_db():
    global _mongo_client
    try:
        if _mongo_client is None:
            _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = _mongo_client[MONGO_DB]
        # Ensure indexes
        db.users.create_index("email", unique=True)
        return db
    except Exception as e:
        print("MongoDB connection failed:", e)
        return None


def _id_str(doc):
    """Convert ObjectId to string in a document."""
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
    return doc


def _ids_str(docs):
    return [_id_str(d) for d in docs]


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _local_ai_fallback(user_msg: str, is_logged_in: bool = False) -> str:
    q = (user_msg or "").lower()
    if any(k in q for k in ["price", "cheapest", "cheap", "deal", "discount"]):
        return (
            "Current top deals: Delhi→Jaipur Bus at Rs.720, Bangalore→Chennai Train at Rs.902, "
            "Delhi→Bangalore Flight at Rs.3,420, and Arijit Singh Concert at Rs.2,400."
        )
    if any(k in q for k in ["book", "how", "steps"]):
        if is_logged_in:
            return "Go to Dashboard, pick a ticket, select seat, complete payment, then download your PDF ticket."
        return "Please login first, then go to Dashboard, choose ticket, select seat, and complete payment."
    if any(k in q for k in ["flight", "bus", "train", "hotel", "concert"]):
        return "We have BUS, TRAIN, FLIGHT, HOTEL, and CONCERT tickets. Open Dashboard to browse all available options."
    return "I can help with prices, routes, and booking steps. Ask for cheapest options or how to book quickly."


# ================= SEED DATA (run once) =================
def seed_sample_data():
    db = get_db()
    if db is None:
        return
    if db.tickets.count_documents({}) == 0:
        db.tickets.insert_many([
            # ── BUS ──────────────────────────────────────────────────────
            {"type": "BUS", "title": "Delhi to Jaipur Volvo AC",        "source": "Delhi",     "destination": "Jaipur",    "price": 800,  "discount": 10, "total_seats": 40,  "created_at": datetime.now()},
            {"type": "BUS", "title": "Mumbai to Goa Sleeper",           "source": "Mumbai",    "destination": "Goa",       "price": 1200, "discount": 0,  "total_seats": 40,  "created_at": datetime.now()},
            {"type": "BUS", "title": "Bangalore to Mysore Express",     "source": "Bangalore", "destination": "Mysore",    "price": 350,  "discount": 5,  "total_seats": 45,  "created_at": datetime.now()},
            {"type": "BUS", "title": "Delhi to Agra Luxury Coach",      "source": "Delhi",     "destination": "Agra",      "price": 600,  "discount": 15, "total_seats": 36,  "created_at": datetime.now()},
            {"type": "BUS", "title": "Hyderabad to Vijayawada Sleeper", "source": "Hyderabad", "destination": "Vijayawada","price": 950,  "discount": 0,  "total_seats": 40,  "created_at": datetime.now()},
            # ── TRAIN ─────────────────────────────────────────────────────
            {"type": "TRAIN", "title": "Patna Rajdhani Express",        "source": "Patna",     "destination": "Delhi",     "price": 1200, "discount": 5,  "total_seats": 60,  "created_at": datetime.now()},
            {"type": "TRAIN", "title": "Bangalore–Chennai Express",     "source": "Bangalore", "destination": "Chennai",   "price": 950,  "discount": 5,  "total_seats": 60,  "created_at": datetime.now()},
            {"type": "TRAIN", "title": "Mumbai Duronto Express",        "source": "Mumbai",    "destination": "Delhi",     "price": 1800, "discount": 10, "total_seats": 72,  "created_at": datetime.now()},
            {"type": "TRAIN", "title": "Shatabdi Express Delhi–Agra",   "source": "Delhi",     "destination": "Agra",      "price": 750,  "discount": 0,  "total_seats": 80,  "created_at": datetime.now()},
            {"type": "TRAIN", "title": "Vande Bharat Delhi–Varanasi",   "source": "Delhi",     "destination": "Varanasi",  "price": 1600, "discount": 8,  "total_seats": 100, "created_at": datetime.now()},
            # ── FLIGHT ────────────────────────────────────────────────────
            {"type": "FLIGHT", "title": "IndiGo 6E-203 Delhi–Mumbai",   "source": "Delhi",     "destination": "Mumbai",    "price": 4500, "discount": 15, "total_seats": 180, "created_at": datetime.now()},
            {"type": "FLIGHT", "title": "Air India AI-101 Delhi–Blr",   "source": "Delhi",     "destination": "Bangalore", "price": 3800, "discount": 10, "total_seats": 150, "created_at": datetime.now()},
            {"type": "FLIGHT", "title": "SpiceJet SG-8169 Mumbai–Goa",  "source": "Mumbai",    "destination": "Goa",       "price": 2800, "discount": 20, "total_seats": 150, "created_at": datetime.now()},
            {"type": "FLIGHT", "title": "Akasa Air QP-1302 Blr–HYD",   "source": "Bangalore", "destination": "Hyderabad", "price": 2200, "discount": 5,  "total_seats": 120, "created_at": datetime.now()},
            {"type": "FLIGHT", "title": "Vistara UK-995 Delhi–Chennai", "source": "Delhi",     "destination": "Chennai",   "price": 5200, "discount": 0,  "total_seats": 160, "created_at": datetime.now()},
            # ── CONCERT ───────────────────────────────────────────────────
            {"type": "CONCERT", "title": "Arijit Singh Live Mumbai",    "source": "Mumbai",    "destination": "Mumbai",    "price": 3000, "discount": 20, "total_seats": 500, "created_at": datetime.now()},
            {"type": "CONCERT", "title": "AP Dhillon World Tour Delhi", "source": "Delhi",     "destination": "Delhi",     "price": 4500, "discount": 10, "total_seats": 800, "created_at": datetime.now()},
            {"type": "CONCERT", "title": "Diljit Dosanjh Dil-Luminati", "source": "Delhi",     "destination": "Delhi",     "price": 2500, "discount": 0,  "total_seats": 1000,"created_at": datetime.now()},
            {"type": "CONCERT", "title": "Nucleya Bass Yatra Bangalore","source": "Bangalore", "destination": "Bangalore", "price": 1200, "discount": 15, "total_seats": 600, "created_at": datetime.now()},
            {"type": "CONCERT", "title": "Sunburn EDM Festival Goa",   "source": "Goa",       "destination": "Goa",       "price": 5500, "discount": 5,  "total_seats": 2000,"created_at": datetime.now()},
            {"type": "CONCERT", "title": "Prateek Kuhad Acoustic Night","source": "Mumbai",    "destination": "Mumbai",    "price": 800,  "discount": 0,  "total_seats": 300, "created_at": datetime.now()},
            # ── HOTEL ─────────────────────────────────────────────────────
            {"type": "HOTEL", "title": "Taj Hotel Mumbai (Deluxe)",     "source": "Mumbai",    "destination": "Mumbai",    "price": 6000, "discount": 10, "total_seats": 100, "created_at": datetime.now()},
            {"type": "HOTEL", "title": "OYO Rooms Delhi Central",       "source": "Delhi",     "destination": "Delhi",     "price": 900,  "discount": 20, "total_seats": 50,  "created_at": datetime.now()},
            {"type": "HOTEL", "title": "The Leela Bangalore",           "source": "Bangalore", "destination": "Bangalore", "price": 8500, "discount": 5,  "total_seats": 80,  "created_at": datetime.now()},
            {"type": "HOTEL", "title": "Zostel Goa Hostel & Stay",      "source": "Goa",       "destination": "Goa",       "price": 700,  "discount": 0,  "total_seats": 30,  "created_at": datetime.now()},
            {"type": "HOTEL", "title": "ITC Grand Chola Chennai",       "source": "Chennai",   "destination": "Chennai",   "price": 7200, "discount": 15, "total_seats": 120, "created_at": datetime.now()},
            {"type": "HOTEL", "title": "Treebo Trend Jaipur",           "source": "Jaipur",    "destination": "Jaipur",    "price": 1500, "discount": 10, "total_seats": 40,  "created_at": datetime.now()},
        ])
        print("Sample tickets seeded.")


# ================= OTP HELPER =================
def generate_otp():
    return str(random.randint(100000, 999999))


# ================= SAVE BOOKING =================
def save_booking_to_db(data):
    db = get_db()
    if db is None:
        return False
    try:
        ticket_id_str = session.get("ticket_id", "")
        booking_doc = {
            "user_id":      data["user_id"],
            "ticket_id":    ticket_id_str,
            "seat_no":      data["seat"],
            "payment_id":   data["booking_id"],
            "final_price":  data["amount"],
            "booking_date": datetime.now(),
        }
        db.bookings.insert_one(booking_doc)
        return True
    except Exception as e:
        print("Save booking error:", e)
        return False


# ================= HOME =================
@app.route("/")
def home():
    return render_template("user/home.html")


# ================= SIGNUP (FIXED: duplicate email check + welcome email) =================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect("/signup")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect("/signup")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect("/signup")

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        db = get_db()
        if db is None:
            flash("Database connection error.", "danger")
            return redirect("/signup")

        # Explicit duplicate-email check (clear UX) + unique index as DB-level safety net
        if db.users.find_one({"email": email}):
            flash("Email already registered. Please login or use a different email.", "warning")
            return redirect("/signup")

        try:
            db.users.insert_one({
                "name":       name,
                "email":      email,
                "password":   hashed,
                "role":       "USER",
                "otp":        None,
                "is_blocked": False,
                "created_at": datetime.now(),
            })

            # Send welcome email (non-blocking — failure doesn't break signup)
            try:
                login_url = request.host_url.rstrip("/") + "/login"
                send_welcome_email(mail, name, email, login_url)
            except Exception as mail_err:
                print(f"Welcome email failed (non-critical): {mail_err}")

            flash("Account created! Check your email for a welcome message.", "success")
            return redirect("/login")

        except DuplicateKeyError:
            flash("Email already registered.", "warning")
            return redirect("/signup")
        except Exception as e:
            flash(f"Signup error: {str(e)}", "danger")
            return redirect("/signup")

    return render_template("user/signup.html", google_oauth_enabled=bool(google_client_id and google_client_secret))


# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect("/dashboard")
    if "admin_id" in session:
        return redirect("/admin/dashboard")

    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        db = get_db()
        if db is None:
            flash("Database connection error", "danger")
            return redirect("/login")

        user = db.users.find_one({"email": email})
        if not user:
            flash("No account found with that email.", "danger")
            return redirect("/login")

        stored_pw = user.get("password")
        if not stored_pw:
            flash("This account uses Google sign-in. Please continue with Google.", "warning")
            return redirect("/login")
        valid = False
        if stored_pw.startswith("$2b$") or stored_pw.startswith("$2a$"):
            try:
                valid = bcrypt.checkpw(password.encode(), stored_pw.encode())
            except Exception:
                valid = False
        else:
            valid = (password == stored_pw)
            if valid:
                new_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                db.users.update_one({"_id": user["_id"]}, {"$set": {"password": new_hash}})

        if not valid:
            flash("Incorrect password. Please try again.", "danger")
            return redirect("/login")

        role = user.get("role", "USER")
        if role == "ADMIN":
            session.clear()
            session["admin_id"]    = str(user["_id"])
            session["admin_name"]  = user["name"]
            session["admin_email"] = user["email"]
            flash(f"Welcome back, {user['name']}! (Admin)", "success")
            return redirect("/admin/dashboard")
        else:
            if user.get("is_blocked"):
                flash("Your account has been blocked. Please contact support.", "danger")
                return redirect("/login")
            session.clear()
            session["user_id"]    = str(user["_id"])
            session["user_name"]  = user["name"]
            session["user_email"] = user["email"]
            # Support ?next= param from search widget / chatbot
            next_url = request.args.get("next", "/dashboard")
            if not next_url.startswith("/"):
                next_url = "/dashboard"
            return redirect(next_url)

    return render_template("user/login.html", google_oauth_enabled=bool(google_client_id and google_client_secret))


@app.route("/auth/google/login")
def auth_google_login():
    if not (google_client_id and google_client_secret):
        flash("Google OAuth is not configured.", "warning")
        return redirect("/login")
    redirect_uri = request.host_url.rstrip("/") + "/auth/google/callback"
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    if not (google_client_id and google_client_secret):
        flash("Google OAuth is not configured.", "warning")
        return redirect("/login")

    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo") or oauth.google.get("https://openidconnect.googleapis.com/v1/userinfo").json()
        email = (userinfo.get("email") or "").strip().lower()
        name = (userinfo.get("name") or "Google User").strip()
        google_sub = userinfo.get("sub", "")
        if not email:
            flash("Google login failed: email not available.", "danger")
            return redirect("/login")

        db = get_db()
        if db is None:
            flash("Database connection error.", "danger")
            return redirect("/login")

        user = db.users.find_one({"email": email})
        if user and user.get("is_blocked"):
            flash("Your account is blocked. Please contact support.", "danger")
            return redirect("/login")

        if not user:
            db.users.insert_one({
                "name": name,
                "email": email,
                "password": None,
                "role": "USER",
                "otp": None,
                "is_blocked": False,
                "auth_provider": "google",
                "google_sub": google_sub,
                "created_at": datetime.now(),
            })
            user = db.users.find_one({"email": email})
        else:
            db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {"name": name, "auth_provider": "google", "google_sub": google_sub}},
            )
            user = db.users.find_one({"_id": user["_id"]})

        session.clear()
        session["user_id"] = str(user["_id"])
        session["user_name"] = user.get("name", name)
        session["user_email"] = user["email"]
        return redirect("/dashboard")
    except Exception as e:
        print("Google OAuth error:", e)
        flash("Google sign-in failed. Please try again.", "danger")
        return redirect("/login")


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    db = get_db()
    tickets     = []
    my_bookings = []

    if db is not None:
        raw_tickets = list(db.tickets.find().sort("_id", DESCENDING))
        for t in raw_tickets:
            t["id"] = str(t["_id"])
            tickets.append(t)

        # Fetch user's bookings with ticket info
        user_id = session["user_id"]
        raw_bookings = list(db.bookings.find({"user_id": user_id}).sort("booking_date", DESCENDING))
        for b in raw_bookings:
            b["id"] = str(b["_id"])
            # Attach ticket details
            try:
                t = db.tickets.find_one({"_id": ObjectId(b["ticket_id"])})
            except Exception:
                t = None
            if t:
                b["title"]       = t.get("title", "")
                b["type"]        = t.get("type", "")
                b["source"]      = t.get("source", "")
                b["destination"] = t.get("destination", "")
            else:
                b["title"]       = "N/A"
                b["type"]        = "N/A"
                b["source"]      = "N/A"
                b["destination"] = "N/A"
            my_bookings.append(b)

    return render_template("user/dashboard.html", tickets=tickets, my_bookings=my_bookings)


# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")          # ← home page after logout


# ================= FORGOT PASSWORD =================
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        db = get_db()
        if db is None:
            flash("Database connection error", "danger")
            return redirect("/forgot-password")

        user = db.users.find_one({"email": email})
        if not user:
            flash("Email not registered", "danger")
            return redirect("/forgot-password")

        otp    = generate_otp()
        expiry = datetime.now() + timedelta(minutes=10)
        db.users.update_one({"email": email}, {"$set": {"otp": otp}})

        try:
            msg = Message(
                "TicketHub Password Reset OTP",
                recipients=[email],
                body=f"Your OTP is {otp}. It expires in 10 minutes."
            )
            mail.send(msg)
        except Exception as e:
            print("Mail error:", e)

        session["reset_email"] = email
        session["reset_otp"]   = otp
        session["otp_expiry"]  = str(expiry)
        flash("OTP sent to your email", "success")
        return redirect("/verify-otp")

    return render_template("user/forgot_password.html")


# ================= VERIFY OTP =================
@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        otp        = request.form.get("otp")
        stored_otp = session.get("reset_otp")
        expiry_str = session.get("otp_expiry")

        if not stored_otp:
            flash("Session expired", "danger")
            return redirect("/forgot-password")

        try:
            expiry = datetime.fromisoformat(expiry_str)
            if datetime.now() > expiry:
                flash("OTP expired", "danger")
                return redirect("/forgot-password")
        except Exception:
            flash("Session error", "danger")
            return redirect("/forgot-password")

        if otp != stored_otp:
            flash("Invalid OTP", "danger")
            return redirect("/verify-otp")

        session["otp_verified"] = True
        return redirect("/reset-password")

    return render_template("user/verify_otp.html")


# ================= RESET PASSWORD =================
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if not session.get("otp_verified"):
        return redirect("/forgot-password")

    if request.method == "POST":
        password = request.form.get("password")
        confirm  = request.form.get("confirm")

        if password != confirm:
            flash("Passwords do not match", "danger")
            return redirect("/reset-password")

        email  = session.get("reset_email")
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        db = get_db()
        if db is not None:
            db.users.update_one({"email": email}, {"$set": {"password": hashed, "otp": None}})

        for k in ["reset_email", "reset_otp", "otp_expiry", "otp_verified"]:
            session.pop(k, None)
        flash("Password reset successful", "success")
        return redirect("/login")

    return render_template("user/reset_password.html")


# ================= SEAT SELECTION =================
@app.route("/seat-selection/<ticket_id>")
def seat_selection(ticket_id):
    if "user_id" not in session:
        return redirect("/login")

    db = get_db()
    ticket       = None
    booked_seats = []

    if db is not None:
        try:
            ticket = db.tickets.find_one({"_id": ObjectId(ticket_id)})
        except Exception:
            ticket = None

        if ticket:
            ticket["id"] = str(ticket["_id"])
            raw_booked = list(db.bookings.find({"ticket_id": ticket_id}, {"seat_no": 1}))
            booked_seats = [b["seat_no"] for b in raw_booked if b.get("seat_no")]

    if not ticket:
        flash("Ticket not found", "danger")
        return redirect("/dashboard")

    session["ticket_id"] = ticket_id
    session["ticket_info"] = {
        "title":       ticket["title"],
        "type":        ticket["type"],
        "source":      ticket["source"],
        "destination": ticket["destination"],
        "price":       float(ticket["price"]),
        "discount":    ticket["discount"],
    }
    return render_template("user/seat_selection.html", ticket=ticket, booked_seats=booked_seats)


@app.route("/select-seat", methods=["POST"])
def select_seat():
    session["selected_seat"] = request.form.get("seat_number")
    return redirect("/payment")


# ================= PAYMENT =================
@app.route("/payment")
def payment():
    if "user_id" not in session:
        return redirect("/login")
    ticket_info = session.get("ticket_info", {})
    seat        = session.get("selected_seat", "N/A")
    price       = ticket_info.get("price", 0)
    discount    = ticket_info.get("discount", 0)
    final_price = round(price - (price * discount / 100), 2)
    return render_template("user/payment.html", seat_number=seat, ticket_info=ticket_info, final_price=final_price)


@app.route("/process-payment", methods=["POST"])
def process_payment():
    session["payment_status"] = "PAID"
    return redirect("/booking-success")


# ================= BOOKING SUCCESS =================
@app.route("/booking-success")
def booking_success():
    if "user_id" not in session:
        return redirect("/login")

    booking_id  = str(uuid.uuid4())[:8].upper()
    ticket_info = session.get("ticket_info", {})
    seat        = session.get("selected_seat", "N/A")
    price       = ticket_info.get("price", 800)
    discount    = ticket_info.get("discount", 0)
    final_price = round(price - (price * discount / 100), 2)

    ticket_title = ticket_info.get('title', ticket_info.get('type', 'Ticket'))
    data = {
        "booking_id":     booking_id,
        "user_id":        session["user_id"],
        "route":          f"{ticket_info.get('source','N/A')} → {ticket_info.get('destination','N/A')}",
        "ticket_type":    f"{ticket_info.get('type','')} - {ticket_title}",
        "seat":           seat,
        "amount":         final_price,
        "payment_status": "PAID",
        # Enriched fields for PDF accuracy
        "source":         ticket_info.get("source", "N/A"),
        "destination":    ticket_info.get("destination", "N/A"),
        "title":          ticket_title,
        "type":           ticket_info.get("type", ""),
        "base_price":     price,
        "discount":       discount,
        "final_price":    final_price,
        "travel_date":    ticket_info.get("travel_date", datetime.now().strftime("%d %b %Y")),
        "ticket_id":      session.get("ticket_id", ""),
    }

    save_booking_to_db(data)
    session["booking_id"]   = booking_id
    session["booking_data"] = data

    # ── Send confirmation email (via email_service module) ──
    try:
        user_email = session.get("user_email", "")
        user_name  = session.get("user_name", "Valued Customer")
        if user_email:
            pdf_buffer = generate_ticket_pdf({**data, "user_name": user_name})
            send_booking_confirmation(mail, user_name, user_email, data, pdf_buffer)
    except Exception as e:
        print(f"Email send error (non-critical): {e}")

    return render_template("user/booking_success.html", **data)


# ================= DOWNLOAD TICKET =================
@app.route("/download-ticket")
def download_ticket():
    if "user_id" not in session:
        return redirect("/login")
    booking_data = session.get("booking_data", {})
    if not booking_data:
        flash("No booking found", "danger")
        return redirect("/dashboard")
    booking_data_with_name = {
        **booking_data,
        "user_name": session.get("user_name", booking_data.get("user_name", "Valued Customer"))
    }
    pdf = generate_ticket_pdf(booking_data_with_name)
    return send_file(pdf, as_attachment=True, download_name=f"ticket_{booking_data.get('booking_id','ticket')}.pdf")


# ================= AI CHAT TEST =================
# ================= AI CHAT TEST =================
@app.route("/api/chat/test")
def chat_test():
    try:
        reply = ai_call([{"role": "user", "content": "Say hello in one sentence."}], max_tokens=60)
        return jsonify({"status": "ok", "model": app.config.get("GROQ_MODEL"), "reply": reply})
    except Exception as e:
        return jsonify({"status": "error", "model": app.config.get("GROQ_MODEL"), "error": str(e)}), 200


# ================= AI CHATBOT API =================
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_messages = data.get("messages")
    if not user_messages and data.get("message"):
        user_messages = [{"role": "user", "content": data.get("message", "")}]
    if not user_messages:
        return jsonify({"error": "Invalid request"}), 400
    is_logged_in  = "user_id" in session
    user_name     = session.get("user_name", "")

    last_user_msg = ""
    for m in reversed(user_messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "").lower()
            break

    booking_keywords    = ["book", "buy", "purchase", "reserve", "ticket", "book a ticket", "get ticket"]
    my_booking_keywords = ["my booking", "my ticket", "view booking", "booking history", "past booking", "show booking"]
    dashboard_keywords  = ["dashboard", "search ticket", "find ticket", "browse ticket"]
    logout_keywords     = ["logout", "sign out", "log out"]

    action = None
    if any(k in last_user_msg for k in booking_keywords):
        action = "book"
    elif any(k in last_user_msg for k in my_booking_keywords):
        action = "my_bookings"
    elif any(k in last_user_msg for k in dashboard_keywords):
        action = "dashboard"
    elif any(k in last_user_msg for k in logout_keywords):
        action = "logout"

    if action and not is_logged_in:
        return jsonify({
            "reply":   "You need to be logged in to do that! Please login or sign up first.",
            "action":  "require_login",
            "buttons": [
                {"label": "🔐 Login",   "url": "/login"},
                {"label": "📝 Sign Up", "url": "/signup"},
            ]
        })

    if action == "book" and is_logged_in:
        return jsonify({
            "reply":   f"Great, {user_name}! What type of ticket would you like to book?",
            "action":  "show_options",
            "buttons": [
                {"label": "🚌 Bus",          "url": "/dashboard"},
                {"label": "🚂 Train",        "url": "/dashboard"},
                {"label": "✈️ Flight",       "url": "/dashboard"},
                {"label": "🏨 Hotel",        "url": "/dashboard"},
                {"label": "🎵 Concert/Event","url": "/dashboard"},
            ]
        })

    if action == "my_bookings" and is_logged_in:
        return jsonify({
            "reply":   f"Sure {user_name}! Here's a quick link to view your bookings.",
            "action":  "show_options",
            "buttons": [{"label": "📋 View My Bookings", "url": "/dashboard#my-bookings"}]
        })

    if action == "dashboard" and is_logged_in:
        return jsonify({
            "reply":   "Head to your dashboard to search and browse all available tickets!",
            "action":  "show_options",
            "buttons": [{"label": "🏠 Go to Dashboard", "url": "/dashboard"}]
        })

    if action == "logout" and is_logged_in:
        return jsonify({
            "reply":   "You can logout using the button below.",
            "action":  "show_options",
            "buttons": [{"label": "🚪 Logout", "url": "/logout"}]
        })

    login_context = ""
    if is_logged_in:
        login_context = f"\n\nThe user is currently logged in as '{user_name}'. Help them book or answer questions."
    else:
        login_context = "\n\nThe user is NOT logged in. If they want to book or do anything requiring an account, tell them to login or sign up."

    messages = [{"role": "system", "content": SYSTEM_PROMPT + login_context}]
    messages.extend(user_messages)

    try:
        reply = ai_call(messages, max_tokens=512, temperature=0.20)
        return jsonify({"reply": reply})
    except Exception as e:
        error_msg = str(e)
        print("AI API error:", error_msg)
        fallback = _local_ai_fallback(last_user_msg, is_logged_in=is_logged_in)
        return jsonify({"reply": fallback}), 200


# ================= ADMIN SETUP =================
@app.route("/admin/setup", methods=["GET", "POST"])
def admin_setup():
    message = None
    if request.method == "POST":
        name     = request.form.get("name", "Admin").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not email or not password:
            message = ("danger", "Email and password are required.")
        else:
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            db = get_db()
            if not db:
                message = ("danger", "Database connection failed.")
            else:
                try:
                    existing = db.users.find_one({"email": email})
                    if existing:
                        db.users.update_one(
                            {"email": email},
                            {"$set": {"password": hashed, "role": "ADMIN", "name": name}}
                        )
                    else:
                        db.users.insert_one({
                            "name":       name,
                            "email":      email,
                            "password":   hashed,
                            "role":       "ADMIN",
                            "otp":        None,
                            "is_blocked": False,
                            "created_at": datetime.now(),
                        })
                    message = ("success", f"Admin account ready! You can now login with: {email}")
                except Exception as e:
                    message = ("danger", f"Error: {str(e)}")

    return render_template("admin/admin_setup.html", message=message)


# ================= ADMIN LOGIN REDIRECT =================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    return redirect("/login")


# ================= ADMIN DASHBOARD =================
@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin_id" not in session:
        return redirect("/admin/login")

    db      = get_db()
    users   = []
    tickets = []
    stats   = {"total_users": 0, "total_tickets": 0, "active_users": 0, "blocked_users": 0}

    if db is not None:
        try:
            raw_users = list(db.users.find({"role": "USER"}).sort("_id", DESCENDING))
            for u in raw_users:
                u["id"]           = str(u["_id"])
                u["ticket_count"] = db.bookings.count_documents({"user_id": u["id"]})
                if "created_at" not in u:
                    u["created_at"] = None
                users.append(u)

            raw_tickets = list(db.tickets.find().sort("_id", DESCENDING))
            for t in raw_tickets:
                t["id"] = str(t["_id"])
                tickets.append(t)

            stats["total_users"]   = db.users.count_documents({"role": "USER"})
            stats["total_tickets"] = db.tickets.count_documents({})
            stats["active_users"]  = db.users.count_documents({"role": "USER", "is_blocked": {"$ne": True}})
            stats["blocked_users"] = db.users.count_documents({"is_blocked": True})
        except Exception as e:
            print("Admin dashboard error:", e)

    return render_template("admin/admin_dashboard.html", users=users, tickets=tickets, stats=stats)


# ================= ADMIN ADD TICKET =================
@app.route("/admin/add-ticket", methods=["GET", "POST"])
def add_ticket():
    if "admin_id" not in session:
        return redirect("/admin/login")
    if request.method == "POST":
        db = get_db()
        if db is not None:
            db.tickets.insert_one({
                "type":        request.form["type"],
                "title":       request.form["title"],
                "source":      request.form["source"],
                "destination": request.form["destination"],
                "price":       float(request.form["price"]),
                "discount":    int(request.form.get("discount", 0)),
                "total_seats": int(request.form.get("total_seats", 40)),
                "created_at":  datetime.now(),
            })
        flash("Ticket added successfully", "success")
        return redirect("/admin/dashboard")
    return render_template("admin/add_ticket.html")


# ================= ADMIN EDIT TICKET =================
@app.route("/admin/edit-ticket/<ticket_id>", methods=["GET", "POST"])
def edit_ticket(ticket_id):
    if "admin_id" not in session:
        return redirect("/admin/login")
    db     = get_db()
    ticket = None
    if db is not None:
        if request.method == "POST":
            try:
                db.tickets.update_one(
                    {"_id": ObjectId(ticket_id)},
                    {"$set": {
                        "type":        request.form["type"],
                        "title":       request.form["title"],
                        "source":      request.form["source"],
                        "destination": request.form["destination"],
                        "price":       float(request.form["price"]),
                        "discount":    int(request.form.get("discount", 0)),
                        "total_seats": int(request.form.get("total_seats", 40)),
                    }}
                )
            except Exception as e:
                print("Edit ticket error:", e)
            flash("Ticket updated", "success")
            return redirect("/admin/dashboard")
        try:
            ticket = db.tickets.find_one({"_id": ObjectId(ticket_id)})
            if ticket:
                ticket["id"] = str(ticket["_id"])
        except Exception:
            ticket = None
    return render_template("admin/edit_ticket.html", ticket=ticket)


# ================= ADMIN DELETE TICKET =================
@app.route("/admin/delete-ticket/<ticket_id>")
def delete_ticket(ticket_id):
    if "admin_id" not in session:
        return redirect("/admin/login")
    db = get_db()
    if db is not None:
        try:
            db.tickets.delete_one({"_id": ObjectId(ticket_id)})
        except Exception as e:
            print("Delete ticket error:", e)
    flash("Ticket deleted", "success")
    return redirect("/admin/dashboard")


# ================= ADMIN BLOCK / UNBLOCK =================
@app.route("/admin/block-user/<user_id>")
def block_user(user_id):
    if "admin_id" not in session:
        return redirect("/admin/login")
    db = get_db()
    if db is not None:
        try:
            db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"is_blocked": True}})
        except Exception as e:
            print("Block user error:", e)
    flash("User blocked", "warning")
    return redirect("/admin/dashboard")


@app.route("/admin/unblock-user/<user_id>")
def unblock_user(user_id):
    if "admin_id" not in session:
        return redirect("/admin/login")
    db = get_db()
    if db is not None:
        try:
            db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"is_blocked": False}})
        except Exception as e:
            print("Unblock user error:", e)
    flash("User unblocked", "success")
    return redirect("/admin/dashboard")


# ================= ADMIN LOGOUT =================
@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect("/")          # ← home page after logout


# ================= TICKETS BY TYPE API =================
@app.route("/api/tickets-by-type")
def tickets_by_type():
    """
    Return TicketHub tickets matching a type (and optionally source/destination).
    Used by the search page 'Book on TicketHub' button to show real bookable tickets.
    """
    db = get_db()
    if db is None:
        return jsonify({"tickets": []})

    ticket_type = request.args.get("type", "").upper().strip()
    source      = request.args.get("source", "").strip().lower()
    destination = request.args.get("destination", "").strip().lower()

    query = {}
    if ticket_type:
        query["type"] = ticket_type

    raw = list(db.tickets.find(query).sort("_id", DESCENDING))
    # CONCERT and HOTEL tickets don't have meaningful source/destination routes
    # so skip location filtering for these types to always show available options
    SKIP_LOCATION_FILTER = {"CONCERT", "HOTEL"}
    tickets = []
    for t in raw:
        # Soft-match source/destination if provided (skip for CONCERT/HOTEL)
        t_src  = t.get("source", "").lower()
        t_dest = t.get("destination", "").lower()
        if ticket_type not in SKIP_LOCATION_FILTER:
            if source and destination:
                # Include if either city matches any field
                if not (source in t_src or source in t_dest or
                        destination in t_src or destination in t_dest):
                    continue
            elif source:
                if not (source in t_src or source in t_dest):
                    continue
            elif destination:
                if not (destination in t_src or destination in t_dest):
                    continue

        discount = t.get("discount", 0)
        price    = float(t.get("price", 0))
        final    = round(price * (1 - discount / 100))
        tickets.append({
            "id":          str(t["_id"]),
            "title":       t.get("title", ""),
            "type":        t.get("type", ""),
            "source":      t.get("source", ""),
            "destination": t.get("destination", ""),
            "price":       price,
            "discount":    discount,
            "final_price": final,
            "total_seats": t.get("total_seats", 0),
        })

    # If no match found with filters, return all tickets of that type
    if not tickets and ticket_type:
        raw2 = list(db.tickets.find({"type": ticket_type}).sort("_id", DESCENDING))
        for t in raw2:
            discount = t.get("discount", 0)
            price    = float(t.get("price", 0))
            final    = round(price * (1 - discount / 100))
            tickets.append({
                "id":          str(t["_id"]),
                "title":       t.get("title", ""),
                "type":        t.get("type", ""),
                "source":      t.get("source", ""),
                "destination": t.get("destination", ""),
                "price":       price,
                "discount":    discount,
                "final_price": final,
                "total_seats": t.get("total_seats", 0),
            })

    return jsonify({"tickets": tickets})


# ================= AI: BEST PLAN COMPARISON =================
@app.route("/ai/best-plan", methods=["GET", "POST"])
def ai_best_plan():
    """
    GET  /ai/best-plan                       — rank all tickets with default prefs
    POST /ai/best-plan  {budget, interests, destination}  — personalised ranking
    """
    db = get_db()
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    # Parse preferences
    if request.method == "POST":
        body = request.get_json() or {}
        budget = _safe_float(body.get("budget", 5000), 5000)
        interests = body.get("interests", ["BUS", "TRAIN", "FLIGHT", "HOTEL", "CONCERT"])
        destination = body.get("destination", "")
    else:
        budget = _safe_float(request.args.get("budget", 5000), 5000)
        interests = request.args.get("interests", "BUS,TRAIN,FLIGHT").split(",")
        destination = request.args.get("destination", "")

    if not isinstance(interests, list):
        interests = ["BUS", "TRAIN", "FLIGHT"]

    prefs = {"budget": budget, "interests": interests, "destination": destination}

    # Load tickets from DB
    raw_tickets = list(db.tickets.find())
    tickets = []
    for t in raw_tickets:
        t["id"] = str(t["_id"])
        tickets.append(t)

    result = get_best_plan(tickets, prefs)
    return jsonify(result)


# ================= AI: TRENDING LOCATIONS =================
@app.route("/ai/trending-locations")
def ai_trending_locations():
    """GET /ai/trending-locations — returns Serper-powered trending tourist spots."""
    db = get_db()
    locations = fetch_trending_locations(db)
    return jsonify({"count": len(locations), "locations": locations})


# ================= AI: RAG CHATBOT (enhanced) =================
@app.route("/api/chat/rag", methods=["POST"])
def chat_rag():
    """RAG-powered chatbot that retrieves website + ticket knowledge before answering."""
    data = request.get_json() or {}
    user_msg = data.get("message", "")
    if not user_msg and data.get("messages"):
        for m in reversed(data.get("messages", [])):
            if m.get("role") == "user" and m.get("content"):
                user_msg = m.get("content", "")
                break
    if not user_msg:
        return jsonify({"error": "Invalid request"}), 400
    is_logged_in = "user_id" in session
    user_name    = session.get("user_name", "")

    db = get_db()
    tickets = []
    if db is not None:
        raw = list(db.tickets.find())
        for t in raw:
            t["id"] = str(t["_id"])
            tickets.append(t)

    # Retrieve relevant knowledge (RAG)
    knowledge_context = rag_retrieve(user_msg, db, tickets)

    # Build enriched system prompt
    rag_system = SYSTEM_PROMPT
    if knowledge_context:
        rag_system += f"\n\n=== RETRIEVED KNOWLEDGE (use this to answer) ===\n{knowledge_context}"
    if is_logged_in:
        rag_system += f"\n\nThe user is logged in as '{user_name}'."
    else:
        rag_system += "\n\nThe user is NOT logged in. Suggest login/signup for booking."

    try:
        reply = ai_call(
            [{"role": "system", "content": rag_system},
             {"role": "user",   "content": user_msg}],
            max_tokens=512, temperature=0.20
        )
        return jsonify({"reply": reply, "rag_used": bool(knowledge_context)})
    except Exception as e:
        print(f"RAG chat error: {e}")
        return jsonify({"reply": _local_ai_fallback(user_msg, is_logged_in=is_logged_in), "rag_used": bool(knowledge_context)}), 200


# ================= SERPER LIVE SEARCH API =================
@app.route("/api/serper-search", methods=["GET", "POST"])
def serper_search_api():
    """
    Live search endpoint powered by Serper (Google Search).
    Accepts GET params or JSON POST body.
    Required: type (bus|flight|hotel|concert|train)
    Optional: source, destination, date, keyword
    """
    if request.method == "POST":
        body = request.get_json() or {}
        ticket_type  = body.get("type", "")
        source       = body.get("source", "").strip()
        destination  = body.get("destination", "").strip()
        date         = body.get("date", "").strip()
        keyword      = body.get("keyword", "").strip()
    else:
        ticket_type  = request.args.get("type", "")
        source       = request.args.get("source", "").strip()
        destination  = request.args.get("destination", "").strip()
        date         = request.args.get("date", "").strip()
        keyword      = request.args.get("keyword", "").strip()

    if not ticket_type:
        return jsonify({"error": "Missing required param: type"}), 400

    results = unified_search(
        ticket_type=ticket_type,
        source=source,
        destination=destination,
        date=date,
        keyword=keyword,
    )

    return jsonify({
        "query": {
            "type":        ticket_type,
            "source":      source,
            "destination": destination,
            "date":        date,
            "keyword":     keyword,
        },
        "count":   len(results),
        "results": results,
    })


# ================= SERPER SEARCH PAGE =================
@app.route("/search")
def search_page():
    """Full-page live search powered by Serper."""
    return render_template("user/search.html")



# ================= TRENDING LOCATIONS PAGE =================
@app.route("/ai/trending-page")
def trending_page():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("user/trending.html")

# ================= RUN =================
if __name__ == "__main__":
    seed_sample_data()
    app.run(host="0.0.0.0", port=app.config.get("PORT", 5000), debug=app.config.get("DEBUG", False))


# ================= 404 HANDLER =================
@app.errorhandler(404)
def page_not_found(e):
    return render_template("user/404.html"), 404


# ================= HOME CHATBOT (for non-logged-in users on home page) =================
@app.route("/api/chat/home", methods=["POST"])
def chat_home():
    """Lightweight chatbot for the home page (no session required)."""
    data = request.get_json() or {}
    if "message" not in data:
        return jsonify({"error": "Invalid request"}), 400

    user_msg = data["message"]
    try:
        reply = ai_call([
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nThe user is browsing the TicketHub home page and is NOT logged in yet."},
            {"role": "user",   "content": user_msg}
        ], max_tokens=400, temperature=0.20)
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"Home chat error: {e}")
        return jsonify({"reply": _local_ai_fallback(user_msg, is_logged_in=False)}), 200
