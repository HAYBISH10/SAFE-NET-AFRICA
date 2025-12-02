import os
import re
import csv
import hashlib
from datetime import datetime

import streamlit as st
import joblib
import pandas as pd
import altair as alt  # for admin bar charts

# ---------- UI CONFIG (MUST BE FIRST STREAMLIT CALL) ----------
st.set_page_config(
    page_title="SAFE-NET AFRICA",
    page_icon="🛡️",
    layout="wide",
)

from utils.utils import (
    predict_with_model,
    explain_phishing,
    explain_toxic,
    guidance_phishing,
    guidance_toxic,
)

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")  # for uploaded screenshots

PHISHING_MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.joblib")
TOXICITY_MODEL_PATH = os.path.join(MODELS_DIR, "toxicity_model.joblib")
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
REPORTS_CSV = os.path.join(DATA_DIR, "reports.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

USER_COLUMNS = [
    "username",
    "full_name",
    "email",
    "phone",         # contact number
    "password_hash",
    "role",
    "email_verified",  # always "yes" for new users
]

REPORT_COLUMNS = [
    "timestamp",
    "username",
    "type",         # phishing / toxicity
    "probability",  # model probability
    "label",        # human-readable label
    "text",         # message text
    "image_path",   # relative path to screenshot (optional)
]

# ---------- USER STORAGE ----------

def init_users_file():
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(USER_COLUMNS)


def read_users_df() -> pd.DataFrame:
    init_users_file()
    df = pd.read_csv(USERS_CSV, dtype=str)
    if df.empty:
        return pd.DataFrame(columns=USER_COLUMNS)
    for col in USER_COLUMNS:
        if col not in df.columns:
            # default values for missing columns
            if col == "email_verified":
                df[col] = "yes"
            else:
                df[col] = ""
    return df[USER_COLUMNS].fillna("")


def write_users_df(df: pd.DataFrame):
    df = df.copy()
    for col in USER_COLUMNS:
        if col not in df.columns:
            if col == "email_verified":
                df[col] = "yes"
            else:
                df[col] = ""
    df = df[USER_COLUMNS]
    df.to_csv(USERS_CSV, index=False, encoding="utf-8")


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def create_user(
    full_name: str,
    username: str,
    email: str,
    phone: str,
    password: str,
    role: str,
):
    df = read_users_df()
    if (df["username"] == username).any():
        return False, "Username already exists."
    if (df["email"] == email).any() and email != "":
        return False, "Email is already registered."
    new_row = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "phone": phone or "",
        "password_hash": hash_password(password),
        "role": role,
        "email_verified": "yes",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    write_users_df(df)
    return True, None


def get_user(username: str):
    df = read_users_df()
    row = df[df["username"] == username]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def update_user(username: str, **fields):
    df = read_users_df()
    mask = df["username"] == username
    if not mask.any():
        return
    for k, v in fields.items():
        if k in USER_COLUMNS:
            df.loc[mask, k] = v
    write_users_df(df)


def update_user_password(username: str, new_password: str):
    df = read_users_df()
    mask = df["username"] == username
    if not mask.any():
        return
    df.loc[mask, "password_hash"] = hash_password(new_password)
    write_users_df(df)


def delete_user(username: str):
    df = read_users_df()
    df = df[df["username"] != username].copy()
    write_users_df(df)


def authenticate(username: str, password: str):
    user = get_user(username)
    if not user:
        return None, "User not found."
    if user["password_hash"] != hash_password(password):
        return None, "Incorrect password."
    return user, None


def ensure_default_admin():
    """
    Ensure there is a default admin account:
    - username: Haybish
    - password: Hassan@57244172
    - email: hassanhaybish@gmail.com
    """
    df = read_users_df()
    if (df["username"] == "Haybish").any():
        return
    create_user(
        full_name="SAFE-NET Admin",
        username="Haybish",
        email="hassanhaybish@gmail.com",
        phone="",  # no phone set initially
        password="Hassan@57244172",
        role="admin",
    )


# Create default admin on startup (idempotent)
ensure_default_admin()

# ---------- REPORT STORAGE (WITH BACKWARD-COMPATIBLE COLUMNS) ----------

def _init_reports_file_if_needed():
    if not os.path.exists(REPORTS_CSV):
        with open(REPORTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REPORT_COLUMNS)


def _ensure_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any existing reports.csv (old format) into the new schema.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    df = df.copy()
    for col in REPORT_COLUMNS:
        if col not in df.columns:
            if col == "probability":
                df[col] = 0.0
            else:
                df[col] = ""
    return df[REPORT_COLUMNS]


def save_report(
    msg_type: str,
    text: str,
    prob: float,
    label: str,
    username: str,
    image_path: str | None = None,
):
    """
    Save one report row to CSV, including optional screenshot path.
    """
    _init_reports_file_if_needed()
    if image_path is None:
        image_path = ""
    with open(REPORTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                username,
                msg_type,
                round(float(prob), 4),
                label,
                text.replace("\n", " ")[:2000],
                image_path,
            ]
        )


def load_my_reports(username: str) -> pd.DataFrame:
    if not os.path.exists(REPORTS_CSV):
        return pd.DataFrame(columns=REPORT_COLUMNS)
    df = pd.read_csv(REPORTS_CSV)
    df = _ensure_report_columns(df)
    return df[df["username"] == username].copy()


def load_all_reports() -> pd.DataFrame:
    if not os.path.exists(REPORTS_CSV):
        return pd.DataFrame(columns=REPORT_COLUMNS)
    df = pd.read_csv(REPORTS_CSV)
    return _ensure_report_columns(df)

# ---------- MODELS ----------

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


phishing_model = load_model(PHISHING_MODEL_PATH)
toxicity_model = load_model(TOXICITY_MODEL_PATH)

# ---------- URL HELPERS ----------

def extract_urls(text: str):
    if not text:
        return []
    return re.findall(r"(https?://[^\s]+)", text)


def url_safety_hints(text: str) -> str:
    urls = extract_urls(text)
    if not urls:
        return "No direct links detected in this message."

    hints = []
    suspicious_tlds = [".ru", ".cn", ".xyz", ".top", ".club"]
    for url in urls:
        u = url.lower()
        url_points = [f"- Detected link: `{url}`"]

        if re.search(r"\d+\.\d+\.\d+\.\d+", u):
            url_points.append(
                "  - Uses a raw IP address, which can be a sign of suspicious links."
            )

        if any(tld in u for tld in suspicious_tlds):
            url_points.append(
                "  - Domain ending looks uncommon for normal services. Stay cautious."
            )

        if "@" in u.split("//")[-1].split("/")[0]:
            url_points.append(
                "  - `@` symbol in the URL can be used to hide the real destination."
            )

        url_points.append(
            "  - Avoid logging in or sharing passwords from this link unless you are 100% sure."
        )

        hints.extend(url_points)

    return "\n".join(hints)

# ---------- SIDEBAR (ONLY LANGUAGE) ----------

def sidebar_info():
    st.sidebar.title("SAFE-NET AFRICA")

    lang = st.sidebar.selectbox(
        "Preferred language (for guidance notes)",
        ["English", "Swahili", "French", "Arabic", "Somali"],
        key="language_select",
    )

    if lang == "Swahili":
        st.sidebar.info(
            "Ujumbe wa usalama utaelekezwa kwa mtumiaji, lakini modeli inatumia Kiingereza."
        )
    elif lang == "French":
        st.sidebar.info(
            "Les conseils sont adaptés, mais le modèle fonctionne surtout en anglais."
        )
    elif lang == "Arabic":
        st.sidebar.info(
            "النصائح مخصصة لك، لكن نموذج الذكاء الاصطناعي يعمل في الغالب باللغة الإنجليزية."
        )
    elif lang == "Somali":
        st.sidebar.info(
            "Talada badbaadadu waa lagu habeyn karaa, laakiin moodelku waxa uu adeegsadaa Ingiriisi."
        )
    else:
        st.sidebar.info(
            "Guidance is tailored for you, but the AI models are trained mainly on English text."
        )

# ---------- AUTH UI HELPERS (USER: LOGIN + REGISTER, ADMIN: LOGIN ONLY) ----------

def user_auth_ui():
    """
    Main page (no ?admin=1):
    - New users register
    - Existing users log in
    """
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
        st.session_state["auth_role"] = None

    if st.session_state["auth_user"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(
                f"Logged in as **{st.session_state['auth_user']}** "
                f"({st.session_state['auth_role']})"
            )
        with col2:
            if st.button("Logout"):
                st.session_state["auth_user"] = None
                st.session_state["auth_role"] = None
                st.rerun()
        return True

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # --- LOGIN TAB ---
    with tab_login:
        st.subheader("Login to SAFE-NET AFRICA")

        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", key="login_button"):
            user, err = authenticate(username, password)
            if err:
                st.error(err)
            else:
                st.session_state["auth_user"] = user["username"]
                st.session_state["auth_role"] = user["role"]
                st.success("Login successful.")
                st.rerun()

    # --- REGISTER TAB ---
    with tab_register:
        st.subheader("Create a new SAFE-NET account")

        full_name = st.text_input("Full name", key="reg_full_name")
        username = st.text_input("Username", key="reg_username")
        email = st.text_input("Email", key="reg_email")
        phone = st.text_input("Contact number", key="reg_phone")
        password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input(
            "Confirm password", type="password", key="reg_confirm_password"
        )

        if st.button("Register", key="register_button"):
            if not full_name or not username or not password or not confirm_password:
                st.warning("Full name, username and passwords are required.")
            elif password != confirm_password:
                st.warning("Passwords do not match.")
            else:
                ok, msg = create_user(
                    full_name=full_name,
                    username=username,
                    email=email,
                    phone=phone,
                    password=password,
                    role="user",
                )
                if ok:
                    st.success("Account created successfully. Please log in from the Login tab.")
                else:
                    st.error(msg)

    return False


def admin_auth_ui():
    """
    Admin page (?admin=1) – login only, no registration.
    """
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
        st.session_state["auth_role"] = None

    if st.session_state["auth_user"]:
        st.success(
            f"Logged in as **{st.session_state['auth_user']}** "
            f"({st.session_state['auth_role']})"
        )
        return True

    st.subheader("Admin Login")

    username = st.text_input("Username", key="admin_login_username")
    password = st.text_input("Password", type="password", key="admin_login_password")

    if st.button("Login", key="admin_login_button"):
        user, err = authenticate(username, password)
        if err:
            st.error(err)
        else:
            st.session_state["auth_user"] = user["username"]
            st.session_state["auth_role"] = user["role"]
            st.success("Login successful.")
            st.rerun()

    return False


def auth_ui():
    """
    (Legacy) Login-only authentication.
    Not used anymore; kept to avoid changing too much structure.
    """
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
        st.session_state["auth_role"] = None

    if st.session_state["auth_user"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(
                f"Logged in as **{st.session_state['auth_user']}** "
                f"({st.session_state['auth_role']})"
            )
        with col2:
            if st.button("Logout"):
                st.session_state["auth_user"] = None
                st.session_state["auth_role"] = None
                st.rerun()
        return True

    st.subheader("Login")

    username = st.text_input("Username", key="legacy_login_username")
    password = st.text_input("Password", type="password", key="legacy_login_password")

    if st.button("Legacy Login", key="legacy_login_button"):
        user, err = authenticate(username, password)
        if err:
            st.error(err)
        else:
            st.session_state["auth_user"] = user["username"]
            st.session_state["auth_role"] = user["role"]
            st.success("Login successful.")
            st.rerun()

    return False

# ---------- PROFILE & DASHBOARD (USER VIEW + THEIR IMAGES + PHONE EDIT) ----------

def user_profile_dashboard(username: str):
    user = get_user(username)
    if not user:
        st.error("User not found.")
        return

    # ---- MINI USER DASHBOARD (NEW, PROFESSIONAL TOUCH) ----
    reports_df = load_my_reports(username)
    total_reports = len(reports_df)
    phishing_count = (reports_df["type"] == "phishing").sum() if not reports_df.empty else 0
    toxicity_count = (reports_df["type"] == "toxicity").sum() if not reports_df.empty else 0

    if not reports_df.empty and "timestamp" in reports_df.columns:
        try:
            last_ts = pd.to_datetime(reports_df["timestamp"], errors="coerce").max()
            last_report_str = last_ts.strftime("%Y-%m-%d %H:%M") if pd.notnull(last_ts) else "—"
        except Exception:
            last_report_str = "—"
    else:
        last_report_str = "—"

    st.markdown("### My Safety Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total reports", total_reports)
    with c2:
        st.metric("Phishing reports", phishing_count)
    with c3:
        st.metric("Abuse reports", toxicity_count)

    st.caption(f"Last report: {last_report_str}")

    st.markdown("---")
    st.subheader("My Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Full name:** {user.get('full_name','')}")
        st.write(f"**Username:** `{user.get('username','')}`")
        st.write(f"**Email:** {user.get('email','')}")
        st.write(f"**Contact number:** {user.get('phone','')}")
    with col2:
        st.write(f"**Role:** {user.get('role','user')}")
        st.write(f"**Email verified:** {user.get('email_verified','yes')}")

    st.markdown("### Update my contact number")

    new_phone = st.text_input(
        "Phone / contact number (optional)",
        value=user.get("phone", ""),
        key="profile_phone",
    )

    if st.button("Save contact number", key="profile_save_phone"):
        update_user(username, phone=new_phone)
        st.success("Contact number updated.")
        st.rerun()

    st.markdown("---")
    st.subheader("My Reports (all types)")
    if reports_df.empty:
        st.info("You have not saved any reports yet.")
    else:
        st.dataframe(reports_df)

        # Show evidence screenshots if any
        if "image_path" in reports_df.columns and reports_df["image_path"].notna().any():
            st.markdown("### My Evidence Screenshots")
            subset = reports_df[reports_df["image_path"].astype(str).str.strip() != ""]
            subset = subset.sort_values("timestamp", ascending=False).head(5)
            for _, row in subset.iterrows():
                img_rel = row["image_path"]
                if isinstance(img_rel, str) and img_rel.strip():
                    img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(BASE_DIR, img_rel)
                    st.caption(f"{row['timestamp']} – {row['type']} – {row['label']}")
                    if os.path.exists(img_abs):
                        st.image(img_abs, use_column_width=True)
                    else:
                        st.text(f"(Image not found: {img_abs})")

# ---------- PHISHING & TOXICITY TABS (WITH USER HISTORY + IMAGE UPLOAD + RISK BAR) ----------

def phishing_tab():
    st.header("🛡️ Digital Literacy – Phishing / Scam Checker")

    st.markdown(
        """
Paste any **SMS, email, DM, or message** that you are unsure about.  
The model will estimate how likely it is to be **spam / phishing**.
        """
    )

    if phishing_model is None:
        st.error(
            "Phishing model is not available. "
            "Please run `SAFE_NET_models.ipynb` to train and save the model "
            "as `models/phishing_model.joblib`."
        )
        return

    current_user = st.session_state.get("auth_user") or "anonymous"

    if "phishing_sample_text" not in st.session_state:
        st.session_state["phishing_sample_text"] = ""

    col_samples = st.columns(2)
    with col_samples[0]:
        if st.button("Use Sample Scam Message", key="phish_sample1"):
            st.session_state["phishing_sample_text"] = (
                "URGENT: Your bank account has been locked. "
                "Click here immediately to verify your details: "
                "http://fake-bank-secure-login.com"
            )
    with col_samples[1]:
        if st.button("Use Sample Safe Message", key="phish_sample2"):
            st.session_state["phishing_sample_text"] = (
                "Hi, just checking if you are still coming to the meeting tomorrow at 2 PM."
            )

    text = st.text_area(
        "Paste the suspicious message here:",
        height=180,
        value=st.session_state["phishing_sample_text"],
        placeholder=(
            "Example: 'Dear user, your account will be closed in 24 hours "
            "unless you verify at http://example-link...'"
        ),
    )

    uploaded_file = st.file_uploader(
        "Optional: upload screenshot of this message (image only)",
        type=["png", "jpg", "jpeg"],
        key="phishing_screenshot",
    )

    if "phishing_last_result" not in st.session_state:
        st.session_state["phishing_last_result"] = None

    if st.button("Analyze Safety", type="primary", key="phishing_button"):
        text_to_check = text.strip()
        if not text_to_check:
            st.warning("Please paste or select a message first.")
            return

        with st.spinner("Analyzing message for phishing risk..."):
            prob, error = predict_with_model(text_to_check, phishing_model)

        if error:
            st.error(error)
            return

        st.session_state["phishing_last_result"] = (text_to_check, prob)

        st.subheader("Result")

        prob_percent = round(prob * 100, 2)

        if prob >= 0.75:
            label = "🚨 Suspicious / Possible Phishing"
            risk_level = "HIGH"
        elif prob >= 0.5:
            label = "⚠️ Some Signs of Phishing – Be Careful"
            risk_level = "MEDIUM"
        else:
            label = "✅ Likely Safe (by model)"
            risk_level = "LOW"

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Model Assessment:**")
            if prob < 0.5:
                st.success(label)
            else:
                st.warning(label)

            st.write(f"**Phishing probability:** `{prob_percent}%`")
            st.write(f"**Risk level:** `{risk_level}`")
            st.progress(min(max(prob, 0.0), 1.0))

        with col2:
            st.write("**Why this result?**")
            st.write(explain_phishing(prob))

        st.markdown("---")
        st.write("### URL Safety Hints")
        st.markdown(url_safety_hints(text_to_check))

        st.markdown("---")
        st.write("### Safety Guidance")
        st.markdown(guidance_phishing(prob))

    if st.button("Save this result for admin review", key="phishing_report_button"):
        last = st.session_state.get("phishing_last_result")
        if not last:
            st.warning("Please analyze a message first before saving.")
        else:
            text_to_check, prob = last
            if prob >= 0.75:
                label = "🚨 Suspicious / Possible Phishing"
            elif prob >= 0.5:
                label = "⚠️ Some Signs of Phishing – Be Careful"
            else:
                label = "✅ Likely Safe (by model)"

            image_path = None
            if uploaded_file is not None:
                ext = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
                safe_user = re.sub(r"[^a-zA-Z0-9_-]", "_", current_user)
                filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_user}{ext}"
                abs_path = os.path.join(EVIDENCE_DIR, filename)
                with open(abs_path, "wb") as out:
                    out.write(uploaded_file.getbuffer())
                image_path = os.path.relpath(abs_path, BASE_DIR)

            save_report("phishing", text_to_check, prob, label, current_user, image_path)
            st.success("This message (and any screenshot) has been saved for admin review.")

    # ----- USER HISTORY FOR PHISHING -----
    st.markdown("---")
    st.subheader("📝 My saved phishing reports")

    if current_user == "anonymous":
        st.info("Log in with your account to see your saved phishing reports.")
    else:
        df = load_my_reports(current_user)
        df_phish = df[df["type"] == "phishing"].copy()
        if df_phish.empty:
            st.info("You have not saved any phishing reports yet.")
        else:
            st.dataframe(df_phish)

            if "image_path" in df_phish.columns and df_phish["image_path"].astype(str).str.strip().any():
                st.markdown("### My phishing report screenshots")
                subset = df_phish[df_phish["image_path"].astype(str).str.strip() != ""]
                subset = subset.sort_values("timestamp", ascending=False).head(5)
                for _, row in subset.iterrows():
                    img_rel = row["image_path"]
                    if isinstance(img_rel, str) and img_rel.strip():
                        img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(BASE_DIR, img_rel)
                        st.caption(f"{row['timestamp']} – {row['label']}")
                        if os.path.exists(img_abs):
                            st.image(img_abs, use_column_width=True)
                        else:
                            st.text(f"(Image not found: {img_abs})")


def toxicity_tab():
    st.header("🤝 Survivor Support – Abuse & Harassment Detector")

    st.markdown(
        """
Paste a **message or comment** you received.  
The model will estimate how likely it is to be **abusive or harmful**.
        """
    )

    if toxicity_model is None:
        st.error(
            "Toxicity model is not available. "
            "Please run `SAFE_NET_models.ipynb` to train and save the model "
            "as `models/toxicity_model.joblib`."
        )
        return

    current_user = st.session_state.get("auth_user") or "anonymous"

    if "toxicity_sample_text" not in st.session_state:
        st.session_state["toxicity_sample_text"] = ""

    col_samples = st.columns(2)
    with col_samples[0]:
        if st.button("Use Sample Abusive Message", key="tox_sample1"):
            st.session_state["toxicity_sample_text"] = (
                "You are useless and nobody wants you here. Just leave."
            )
    with col_samples[1]:
        if st.button("Use Sample Supportive Message", key="tox_sample2"):
            st.session_state["toxicity_sample_text"] = (
                "Thank you for sharing your opinion, I really appreciate your honesty."
            )

    text = st.text_area(
        "Paste the message here:",
        height=180,
        value=st.session_state["toxicity_sample_text"],
        placeholder="Example: A hurtful or aggressive comment that you received online.",
    )

    uploaded_file = st.file_uploader(
        "Optional: upload screenshot of this message (image only)",
        type=["png", "jpg", "jpeg"],
        key="toxicity_screenshot",
    )

    if "toxicity_last_result" not in st.session_state:
        st.session_state["toxicity_last_result"] = None

    if st.button("Check for Abuse", type="primary", key="toxicity_button"):
        text_to_check = text.strip()
        if not text_to_check:
            st.warning("Please paste or select a message first.")
            return

        with st.spinner("Checking for abusive or harmful language..."):
            prob, error = predict_with_model(text_to_check, toxicity_model)

        if error:
            st.error(error)
            return

        st.session_state["toxicity_last_result"] = (text_to_check, prob)

        st.subheader("Result")

        prob_percent = round(prob * 100, 2)

        if prob >= 0.75:
            label = "🚨 Abusive / Harmful (by model)"
            risk_level = "HIGH"
        elif prob >= 0.5:
            label = "⚠️ Some Signs of Abusive Language"
            risk_level = "MEDIUM"
        else:
            label = "✅ Not Strongly Abusive (by model)"
            risk_level = "LOW"

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Model Assessment:**")
            if prob < 0.5:
                st.success(label)
            else:
                st.warning(label)

            st.write(f"**Abuse probability:** `{prob_percent}%`")
            st.write(f"**Risk level:** `{risk_level}`")
            st.progress(min(max(prob, 0.0), 1.0))

        with col2:
            st.write("**Why this result?**")
            st.write(explain_toxic(prob))

        st.markdown("---")
        st.write("### Supportive Guidance")
        st.markdown(guidance_toxic(prob))

    if st.button("Save this result for admin review", key="toxicity_report_button"):
        last = st.session_state.get("toxicity_last_result")
        if not last:
            st.warning("Please check a message first before saving.")
        else:
            text_to_check, prob = last
            if prob >= 0.75:
                label = "🚨 Abusive / Harmful (by model)"
            elif prob >= 0.5:
                label = "⚠️ Some Signs of Abusive Language"
            else:
                label = "✅ Not Strongly Abusive (by model)"

            image_path = None
            if uploaded_file is not None:
                ext = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
                safe_user = re.sub(r"[^a-zA-Z0-9_-]", "_", current_user)
                filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_user}{ext}"
                abs_path = os.path.join(EVIDENCE_DIR, filename)
                with open(abs_path, "wb") as out:
                    out.write(uploaded_file.getbuffer())
                image_path = os.path.relpath(abs_path, BASE_DIR)

            save_report("toxicity", text_to_check, prob, label, current_user, image_path)
            st.success("This message (and any screenshot) has been saved for admin review.")

    # ----- USER HISTORY FOR TOXICITY -----
    st.markdown("---")
    st.subheader("📝 My saved abuse/harassment reports")

    if current_user == "anonymous":
        st.info("Log in with your account to see your saved abuse reports.")
    else:
        df = load_my_reports(current_user)
        df_tox = df[df["type"] == "toxicity"].copy()
        if df_tox.empty:
            st.info("You have not saved any abuse/harassment reports yet.")
        else:
            st.dataframe(df_tox)

            if "image_path" in df_tox.columns and df_tox["image_path"].astype(str).str.strip().any():
                st.markdown("### My abuse/harassment screenshots")
                subset = df_tox[df_tox["image_path"].astype(str).str.strip() != ""]
                subset = subset.sort_values("timestamp", ascending=False).head(5)
                for _, row in subset.iterrows():
                    img_rel = row["image_path"]
                    if isinstance(img_rel, str) and img_rel.strip():
                        img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(BASE_DIR, img_rel)
                        st.caption(f"{row['timestamp']} – {row['label']}")
                        if os.path.exists(img_abs):
                            st.image(img_abs, use_column_width=True)
                        else:
                            st.text(f"(Image not found: {img_abs})")

# ---------- ADMIN PORTAL (ANALYTICS + REPORT IMAGE VIEW) ----------

def admin_analytics_section():
    st.subheader("📊 Analytics Dashboard")

    df_reports = load_all_reports()
    if df_reports.empty:
        st.info("No reports yet. Analytics will appear when reports are saved.")
        return

    df_reports["probability"] = pd.to_numeric(
        df_reports["probability"], errors="coerce"
    ).fillna(0.0)

    total_reports = len(df_reports)
    very_harmful = (df_reports["probability"] >= 0.75).sum()
    medium_risk = (
        (df_reports["probability"] >= 0.5) & (df_reports["probability"] < 0.75)
    ).sum()
    ok_safe = (df_reports["probability"] < 0.5).sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div style="background-color:#ff4d4d;padding:16px;border-radius:10px;color:white;">
                <h4 style="margin:0;">Very Harmful</h4>
                <p style="font-size:28px;margin:0;"><b>{very_harmful}</b></p>
                <p style="margin:0;font-size:12px;">Reports with high risk (prob ≥ 0.75)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background-color:#2ecc71;padding:16px;border-radius:10px;color:white;">
                <h4 style="margin:0;">Medium Risk</h4>
                <p style="font-size:28px;margin:0;"><b>{medium_risk}</b></p>
                <p style="margin:0;font-size:12px;">Reports with some risk (0.5 ≤ prob &lt; 0.75)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color:#3498db;padding:16px;border-radius:10px;color:white;">
                <h4 style="margin:0;">Likely OK</h4>
                <p style="font-size:28px;margin:0;"><b>{ok_safe}</b></p>
                <p style="margin:0;font-size:12px;">Reports considered low risk (prob &lt; 0.5)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(f"Total reports: {total_reports}")

    # Bar chart – risk level distribution
    st.markdown("#### Risk level distribution")
    risk_data = pd.DataFrame(
        {
            "Risk level": ["High", "Medium", "Low"],
            "Count": [very_harmful, medium_risk, ok_safe],
        }
    )
    chart_risk = (
        alt.Chart(risk_data)
        .mark_bar()
        .encode(
            x=alt.X("Risk level:N", title="Risk level"),
            y=alt.Y("Count:Q", title="Number of reports"),
            color=alt.Color("Risk level:N"),
        )
    )
    st.altair_chart(chart_risk, use_container_width=True)

    # Bar chart – reports by type (phishing vs toxicity)
    st.markdown("#### Reports by type")
    type_counts = (
        df_reports["type"].fillna("unknown").value_counts().reset_index()
    )
    type_counts.columns = ["Type", "Count"]
    chart_type = (
        alt.Chart(type_counts)
        .mark_bar()
        .encode(
            x=alt.X("Type:N", title="Report type"),
            y=alt.Y("Count:Q", title="Number of reports"),
            color=alt.Color("Type:N"),
        )
    )
    st.altair_chart(chart_type, use_container_width=True)

    # NEW: reports over time (date-based)
    st.markdown("#### Reports over time")
    try:
        df_time = df_reports.copy()
        df_time["date"] = pd.to_datetime(df_time["timestamp"], errors="coerce").dt.date
        df_time = df_time.dropna(subset=["date"])
        if not df_time.empty:
            time_counts = df_time.groupby("date").size().reset_index(name="Count")
            chart_time = (
                alt.Chart(time_counts)
                .mark_bar()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Count:Q", title="Number of reports"),
                )
            )
            st.altair_chart(chart_time, use_container_width=True)
        else:
            st.caption("Not enough valid timestamp data to plot over time.")
    except Exception:
        st.caption("Could not compute time-based chart (timestamp format issue).")


def admin_portal(username: str):
    st.header("🛠 Admin Portal")

    user = get_user(username)
    if not user or user.get("role") != "admin":
        st.error("You are not authorized to view this page.")
        return

    st.subheader("My Admin Profile")
    st.write(f"**Full name:** {user.get('full_name','')}")
    st.write(f"**Username:** `{user.get('username','')}`")
    st.write(f"**Email:** {user.get('email','')}")
    st.write(f"**Contact number:** {user.get('phone','')}")
    st.write(f"**Email verified:** {user.get('email_verified','yes')}")

    # NEW: Model & thresholds info
    with st.expander("Model & risk thresholds (for documentation)", expanded=False):
        st.markdown(
            """
- **Phishing model:** TF-IDF + Logistic Regression (`phishing_model.joblib`)  
- **Abuse model:** TF-IDF + Logistic Regression (`toxicity_model.joblib`)  

**Risk bands (both models):**

- **High risk:** probability ≥ 0.75  
- **Medium risk:** 0.5 ≤ probability &lt; 0.75  
- **Low risk:** probability &lt; 0.5  

These thresholds are applied consistently in both the user UI and the admin analytics.
            """
        )

    st.markdown("---")

    # ---- Analytics Dashboard ----
    admin_analytics_section()

    st.markdown("---")
    st.subheader("All Users")
    df_users = read_users_df()
    if df_users.empty:
        st.info("No users registered yet.")
    else:
        st.dataframe(df_users)

    # ---- Create New User (by Admin) ----
    st.subheader("➕ Create new user")

    with st.form("admin_create_user_form"):
        new_full_name = st.text_input("Full name", key="admin_new_full_name")
        new_username = st.text_input("Username", key="admin_new_username")
        new_email = st.text_input("Email", key="admin_new_email")
        new_phone = st.text_input("Contact number (optional)", key="admin_new_phone")
        new_password = st.text_input(
            "Password", type="password", key="admin_new_password"
        )
        new_role = st.selectbox("Role", ["user", "admin"], key="admin_new_role")
        submitted = st.form_submit_button("Create user")

    if submitted:
        if not new_full_name or not new_username or not new_password:
            st.warning("Full name, username, and password are required.")
        else:
            ok, msg = create_user(
                full_name=new_full_name,
                username=new_username,
                email=new_email,
                phone=new_phone,
                password=new_password,
                role=new_role,
            )
            if ok:
                st.success(f"User `{new_username}` created.")
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.subheader("Manage Existing Users")

    if df_users.empty:
        st.info("No users to manage.")
        return

    usernames = df_users["username"].tolist()
    selected_username = st.selectbox(
        "Select a user", usernames, key="admin_user_select"
    )

    if selected_username:
        selected_user = get_user(selected_username)
        st.write(f"**Editing user:** `{selected_username}`")

        col1, col2 = st.columns(2)
        with col1:
            edit_full_name = st.text_input(
                "Full name",
                value=selected_user.get("full_name", ""),
                key="admin_edit_full_name",
            )
            edit_email = st.text_input(
                "Email",
                value=selected_user.get("email", ""),
                key="admin_edit_email",
            )
            edit_phone = st.text_input(
                "Contact number",
                value=selected_user.get("phone", ""),
                key="admin_edit_phone",
            )
        with col2:
            edit_role = st.selectbox(
                "Role",
                ["user", "admin"],
                index=0 if selected_user.get("role", "user") == "user" else 1,
                key="admin_edit_role",
            )
            st.write(f"Email verified: {selected_user.get('email_verified','yes')}")

        new_password = st.text_input(
            "Set new password (leave blank to keep current)",
            type="password",
            key="admin_edit_password",
        )

        colb1, colb2, colb3, colb4 = st.columns(4)
        with colb1:
            if st.button("Update user", key="admin_update_user"):
                update_user(
                    selected_username,
                    full_name=edit_full_name,
                    email=edit_email,
                    phone=edit_phone,
                    role=edit_role,
                )
                if new_password:
                    update_user_password(selected_username, new_password)
                st.success("User updated.")
                st.rerun()
        with colb2:
            if st.button("Make admin", key="admin_make_admin"):
                update_user(selected_username, role="admin")
                st.success(f"User `{selected_username}` is now an admin.")
                st.rerun()
        with colb3:
            if selected_username == username:
                st.info("You cannot delete your own admin account from here.")
            else:
                if st.button("Delete user", key="admin_delete_user"):
                    delete_user(selected_username)
                    st.success(f"User `{selected_username}` deleted.")
                    st.rerun()
        with colb4:
            st.empty()  # spacing

    st.markdown("---")
    st.subheader("All Reports (raw data)")
    df_reports = load_all_reports()
    if df_reports.empty:
        st.info("No reports saved yet.")
    else:
        st.dataframe(df_reports)

        if "image_path" in df_reports.columns and df_reports["image_path"].astype(str).str.strip().any():
            st.markdown("### Report screenshots (latest)")
            subset = df_reports[df_reports["image_path"].astype(str).str.strip() != ""]
            subset = subset.sort_values("timestamp", ascending=False).head(10)
            for _, row in subset.iterrows():
                img_rel = row["image_path"]
                if isinstance(img_rel, str) and img_rel.strip():
                    img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(BASE_DIR, img_rel)
                    st.caption(
                        f"{row['timestamp']} – {row['username']} – {row['type']} – {row['label']}"
                    )
                    if os.path.exists(img_abs):
                        st.image(img_abs, use_column_width=True)
                    else:
                        st.text(f"(Image not found: {img_abs})")

# ---------- MAIN ----------

def main():
    sidebar_info()

    st.title("SAFE-NET AFRICA")
    st.markdown(
        """
**AI-Powered Digital Safety & Survivor Support for Women & Girls in Africa**

Use this tool to check if messages look like **phishing / scams** or **abusive / harmful**,
and to save reports with optional screenshots for follow-up and support.
        """
    )

    # NEW: How it works (professional product-style explanation)
    with st.expander("How SAFE-NET AFRICA works", expanded=False):
        st.markdown(
            """
1. **Create an account or log in**  
   - Your profile keeps your reports and contact number in one place.

2. **Paste a message or upload a screenshot**  
   - Use the **Phishing / Scam Checker** for suspicious links or financial requests.  
   - Use the **Abuse & Harassment Detector** for hurtful or threatening messages.

3. **Review the risk level & guidance**  
   - The system estimates **risk probability** and shows **safety tips**.

4. **Save reports (optional)**  
   - Saved reports appear in **your dashboard** and in the **admin portal** for follow-up.  
   - Screenshots are stored as evidence attached to each report.
            """
        )

    with st.expander("Safety-by-Design & Privacy", expanded=True):
        st.markdown(
            """
- We use **accounts** so you can see your own profile, reports and contact number.
- We store:
  - Your **account details** (name, username, email, contact number, role)
  - The **messages you choose to save** and optional screenshots
- We do **not** use third-party tracking scripts or external analytics.
            """
        )

    # ---- Check query params to see if admin page is requested ----
    try:
        qp = st.query_params  # Streamlit modern API
        raw_admin = qp.get("admin", "0")
        if isinstance(raw_admin, list):
            admin_flag = raw_admin[0]
        else:
            admin_flag = raw_admin
    except Exception:
        admin_flag = "0"

    is_admin_page = admin_flag == "1"

    if is_admin_page:
        # ----- ADMIN PAGE (/?admin=1) -----
        logged_in = admin_auth_ui()
        if not logged_in:
            return

        username = st.session_state["auth_user"]
        role = st.session_state["auth_role"]

        if role != "admin":
            st.error("You are not authorized to access the admin portal.")
            return

        admin_portal(username)
        return

    # ----- USER PAGE (/) -----
    logged_in = user_auth_ui()
    if not logged_in:
        return

    username = st.session_state["auth_user"]
    role = st.session_state["auth_role"]

    # If admin logs in on user page, show hint for admin URL
    if role == "admin":
        st.info(
            "You are logged in as **ADMIN**. "
            "To manage users and view analytics, open: `http://localhost:8519/?admin=1`"
        )

    # ---- Main SAFE-NET tools ----
    st.markdown("---")
    user_profile_dashboard(username)
    st.markdown("---")

    tab1, tab2 = st.tabs(
        [
            "🛡️ Digital Literacy – Phishing / Scam Checker",
            "🤝 Survivor Support – Abuse & Harassment Detector",
        ]
    )

    with tab1:
        phishing_tab()
    with tab2:
        toxicity_tab()


if __name__ == "__main__":
    main()
