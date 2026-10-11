import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse

st.set_page_config(
    page_title="PhishGuard | Phishing URL Detection",
    page_icon="🛡️",
    layout="centered"
)

# terminal-style CSS
# unsafe_allow_html=True lets us inject raw CSS (streamlit strips html by default)
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Courier New', monospace !important;
}

.stTextInput > div > div > input {
    background-color: #161b22 !important;
    border: 0.5px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: monospace !important;
    border-radius: 6px !important;
}

.stButton > button {
    background-color: #238636 !important;
    color: white !important;
    border: none !important;
    font-family: monospace !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 0.4rem 1.2rem !important;
}
.stButton > button:hover {
    background-color: #2ea043 !important;
}

.terminal-card {
    background-color: #161b22;
    border: 0.5px solid #21262d;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}

.terminal-topbar {
    background-color: #0d1117;
    border-bottom: 0.5px solid #21262d;
    border-radius: 8px 8px 0 0;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    gap: 6px;
    margin: -16px -16px 14px -16px;
}
.dot-red   { display:inline-block; width:10px; height:10px; border-radius:50%; background:#f85149; }
.dot-yellow{ display:inline-block; width:10px; height:10px; border-radius:50%; background:#f0883e; }
.dot-green { display:inline-block; width:10px; height:10px; border-radius:50%; background:#30a14e; }
.term-title{ font-size:11px; color:#8b949e; margin-left:8px; }

.result-danger {
    background-color: #160b0b;
    border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.result-safe {
    background-color: #0b160b;
    border: 1px solid #30a14e;
    border-left: 4px solid #30a14e;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 10px;
    background: #0d1117;
    border-radius: 4px;
    border: 0.5px solid #21262d;
    margin-bottom: 4px;
    font-size: 12px;
}
.feat-key { color: #8b949e; }
.feat-bad  { color: #f85149; font-weight: 500; }
.feat-ok   { color: #30a14e; font-weight: 500; }
.feat-mid  { color: #f59e0b; font-weight: 500; }

.check-item {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 0.5px solid #21262d;
    font-size: 12px;
}
.check-label { color: #8b949e; }
.check-fail  { color: #f85149; font-weight: 500; }
.check-pass  { color: #30a14e; font-weight: 500; }
.check-warn  { color: #f59e0b; font-weight: 500; }

.threat-tag {
    display: inline-block;
    padding: 1px 7px;
    background: #2b0e0e;
    border: 0.5px solid #f85149;
    border-radius: 3px;
    font-size: 10px;
    color: #f85149;
    margin-left: 6px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# --- load models (cached so they only load once at startup) ---
@st.cache_resource
def load_models():
    import tensorflow as tf
    rf_model  = joblib.load("rf_model.joblib")
    dt_model  = joblib.load("dt_model.joblib")
    ann_model = tf.keras.models.load_model("ann_model.keras")
    scaler    = joblib.load("scaler.joblib")
    return rf_model, dt_model, ann_model, scaler

rf_model, dt_model, ann_model, scaler = load_models()

# --- scan limit per session ---
MAX_SCANS = 15
if "scan_count" not in st.session_state:
    try:
        import time
        stored_count = int(st.query_params.get("sc", 0))
        stored_ts    = float(st.query_params.get("sc_ts", 0))
        if time.time() - stored_ts > 86400:
            st.session_state["scan_count"] = 0
            st.query_params.pop("sc",    None)
            st.query_params.pop("sc_ts", None)
        else:
            st.session_state["scan_count"] = min(max(stored_count, 0), MAX_SCANS)
    except (ValueError, TypeError):
        st.session_state["scan_count"] = 0


# --- input validation ---
def validate_url(url: str):
    """
    Returns (is_valid, error_message).
    Blocks: non-http(s) protocols, localhost, private IPs, URLs that are too long.
    """
    # must be http or https
    if not url.startswith(("http://", "https://")):
        return False, "URL must start with http:// or https://"

    # sanity check on length
    if len(url) > 2000:
        return False, f"URL is too long ({len(url)} chars). Max allowed: 2000."

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # block loopback / localhost
    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return False, "Scanning localhost is not allowed."

    # block private IP ranges
    if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)", hostname):
        return False, "Scanning private network addresses is not supported."

    # block missing hostname entirely
    if not hostname:
        return False, "Could not parse a valid hostname from the URL."

    return True, ""


# --- feature extraction ---
def extract_features(url):
    import requests
    from bs4 import BeautifulSoup

    parsed = urlparse(url)
    url_length = len(url)
    is_https = 1 if url.startswith("https") else 0
    no_of_digits = sum(c.isdigit() for c in url)
    domain_parts = parsed.netloc.split(".")
    no_of_subdomain = max(len(domain_parts) - 2, 0)
    is_domain_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.netloc) else 0
    tld = domain_parts[-1] if domain_parts else ""
    tld_length = len(tld)
    no_of_letters = sum(c.isalpha() for c in url)
    letter_ratio = no_of_letters / url_length if url_length > 0 else 0
    digit_ratio = no_of_digits / url_length if url_length > 0 else 0
    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")
    no_of_other_special = sum(1 for c in url if c in ["@", "!", "#", "$", "%", "^", "*"])
    special_char_ratio = no_of_other_special / url_length if url_length > 0 else 0
    domain_length = len(parsed.netloc)

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = response.text.lower()

        no_of_image = len(soup.find_all("img"))
        no_of_js = len(soup.find_all("script"))
        no_of_css = len(soup.find_all("link", rel="stylesheet"))
        line_of_code = len(response.text.split("\n"))
        largest_line = max((len(l) for l in response.text.split("\n")), default=0)
        no_of_self_ref = len([a for a in soup.find_all("a", href=True)
                               if parsed.netloc in a["href"]])
        no_of_empty_ref = len([a for a in soup.find_all("a", href=True)
                                if a["href"] in ["#", "", "/"]])
        no_of_external_ref = len([a for a in soup.find_all("a", href=True)
                                   if a["href"].startswith("http")
                                   and parsed.netloc not in a["href"]])

        has_title = 1 if soup.find("title") else 0
        has_favicon = 1 if soup.find("link", rel=lambda r: r and "icon" in r) else 0
        has_description = 1 if soup.find("meta", attrs={"name": "description"}) else 0
        is_responsive = 1 if soup.find("meta", attrs={"name": "viewport"}) else 0
        robots = 1 if soup.find("meta", attrs={"name": "robots"}) else 0
        has_submit = 1 if soup.find("input", type="submit") or soup.find("button") else 0
        has_hidden = 1 if soup.find("input", type="hidden") else 0
        has_password = 1 if soup.find("input", type="password") else 0
        no_of_iframe = len(soup.find_all("iframe"))
        no_of_popup = page_text.count("window.open")
        has_ext_form = 1 if any(parsed.netloc not in f.get("action", "")
                                 for f in soup.find_all("form", action=True)) else 0
        social_keywords = ["facebook.com", "twitter.com", "instagram.com",
                           "linkedin.com", "youtube.com"]
        has_social = 1 if any(k in page_text for k in social_keywords) else 0
        has_copyright = 1 if "©" in page_text or "copyright" in page_text else 0
        bank = 1 if any(k in page_text for k in ["bank", "banking"]) else 0
        pay = 1 if any(k in page_text for k in ["paypal", "payment", "pay now"]) else 0
        crypto = 1 if any(k in page_text for k in ["bitcoin", "crypto", "ethereum"]) else 0

        # accessible pages get URLSimilarityIndex = 100 (matches training data)
        url_similarity = 100

        try:
            le = joblib.load("le.joblib")
            tld_encoded = int(le.transform([tld])[0]) if tld in le.classes_ else 0
        except:
            tld_encoded = 0

        char_continuation = len(max(re.findall(r"[a-z]+", url), key=len, default="")) / url_length if url_length > 0 else 0
        url_char_prob = no_of_letters / url_length if url_length > 0 else 0
        obfuscation = 1 if "%" in url else 0
        no_of_obfuscated = url.count("%")
        obfuscation_ratio = no_of_obfuscated / url_length if url_length > 0 else 0

    except Exception:
        # can't fetch the page — fall back to URL-only features
        # use a lower URLSimilarityIndex if the URL looks suspicious
        _suspicious = (
            not url.startswith("https") or
            bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url)) or
            any(c in url for c in ["-", "@", "%"]) or
            any(kw in url.lower() for kw in ["login", "verify", "secure", "update",
                                              "account", "signin", "bank", "paypal"]) or
            any(url.lower().endswith(t) or (t + "/") in url.lower()
                for t in [".tk", ".ml", ".ga", ".cf", ".gq"])
        )
        url_similarity = 10 if _suspicious else 80
        line_of_code = largest_line = has_title = has_favicon = has_description = 0
        is_responsive = robots = has_submit = has_hidden = has_password = 0
        no_of_iframe = no_of_popup = has_ext_form = has_social = has_copyright = 0
        bank = pay = crypto = no_of_image = no_of_css = no_of_js = 0
        no_of_self_ref = no_of_empty_ref = no_of_external_ref = 0
        char_continuation = len(max(re.findall(r"[a-z]+", url), key=len, default="")) / url_length if url_length > 0 else 0
        url_char_prob = no_of_letters / url_length if url_length > 0 else 0
        obfuscation = 1 if "%" in url else 0
        no_of_obfuscated = url.count("%")
        obfuscation_ratio = no_of_obfuscated / url_length if url_length > 0 else 0
        try:
            le = joblib.load("le.joblib")
            tld_encoded = int(le.transform([tld])[0]) if tld in le.classes_ else 0
        except:
            tld_encoded = 0

    # must match the exact column order used during scaler training
    # TLD_encoded goes last (same as in the notebook)
    features = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "URLSimilarityIndex": url_similarity,
        "CharContinuationRate": char_continuation,
        "TLDLegitimateProb": 0.8 if tld in ["com", "org", "net", "edu", "gov"] else 0.3,
        "URLCharProb": url_char_prob,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special,
        "SpacialCharRatioInURL": special_char_ratio,
        "IsHTTPS": is_https,
        "LineOfCode": line_of_code,
        "LargestLineLength": largest_line,
        "HasTitle": has_title,
        "DomainTitleMatchScore": 0,
        "URLTitleMatchScore": 0,
        "HasFavicon": has_favicon,
        "Robots": robots,
        "IsResponsive": is_responsive,
        "NoOfURLRedirect": 0,
        "NoOfSelfRedirect": 0,
        "HasDescription": has_description,
        "NoOfPopup": no_of_popup,
        "NoOfiFrame": no_of_iframe,
        "HasExternalFormSubmit": has_ext_form,
        "HasSocialNet": has_social,
        "HasSubmitButton": has_submit,
        "HasHiddenFields": has_hidden,
        "HasPasswordField": has_password,
        "Bank": bank,
        "Pay": pay,
        "Crypto": crypto,
        "HasCopyrightInfo": has_copyright,
        "NoOfImage": no_of_image,
        "NoOfCSS": no_of_css,
        "NoOfJS": no_of_js,
        "NoOfSelfRef": no_of_self_ref,
        "NoOfEmptyRef": no_of_empty_ref,
        "NoOfExternalRef": no_of_external_ref,
        "TLD_encoded": tld_encoded,  # must be last
    }
    return features


def predict(url: str):
    # returns phishing probability for all 3 models + final verdict
    # predict_proba gives [P(safe), P(phishing)] — we want the phishing one
    feats = extract_features(url)
    if feats is None:
        return None, None, None, "ERROR", None

    feat_values = list(feats.values())
    while len(feat_values) < 51:
        feat_values.append(0)
    feat_values = feat_values[:51]

    X = np.array(feat_values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # label 0 = Phishing, 1 = Legitimate in training data
    rf_phishing_idx = list(rf_model.classes_).index(0)
    rf_prob = rf_model.predict_proba(X_scaled)[0][rf_phishing_idx]

    dt_phishing_idx = list(dt_model.classes_).index(0)
    dt_prob = dt_model.predict_proba(X_scaled)[0][dt_phishing_idx]

    # ANN sigmoid output = P(Legitimate), so 1 - that = P(Phishing)
    ann_prob = 1.0 - float(ann_model.predict(X_scaled, verbose=0)[0][0])

    # RF + ANN must both agree before flagging as phishing
    # this reduces false positives on legit sites that have login/payment forms
    rf_says_phishing  = rf_prob  >= 0.5
    ann_says_phishing = ann_prob >= 0.5
    verdict = "PHISHING" if (rf_says_phishing and ann_says_phishing) else "SAFE"
    return rf_prob, dt_prob, ann_prob, verdict, feats


def get_security_checks(url: str, feats: dict) -> list:
    checks = []

    if feats.get("IsHTTPS", 0) == 1:
        checks.append(("HTTPS Protocol", "✓ SECURE", "pass"))
    else:
        checks.append(("HTTPS Protocol", "✗ NOT SECURE", "fail"))

    url_len = feats.get("URLLength", 0)
    if url_len > 75:
        checks.append(("URL Length", f"✗ {url_len} chars (LONG)", "fail"))
    elif url_len > 50:
        checks.append(("URL Length", f"⚠ {url_len} chars", "warn"))
    else:
        checks.append(("URL Length", f"✓ {url_len} chars", "pass"))

    sub = feats.get("NoOfSubDomain", 0)
    if sub >= 3:
        checks.append(("Subdomains", f"✗ {sub} levels (HIGH)", "fail"))
    elif sub == 2:
        checks.append(("Subdomains", f"⚠ {sub} levels", "warn"))
    else:
        checks.append(("Subdomains", f"✓ {sub} level", "pass"))

    if feats.get("IsDomainIP", 0) == 1:
        checks.append(("Domain (IP?)", "✗ IP AS DOMAIN", "fail"))
    else:
        checks.append(("Domain (IP?)", "✓ Normal domain", "pass"))

    if feats.get("HasPasswordField", 0) == 1:
        checks.append(("Password Field", "✗ FOUND", "fail"))
    else:
        checks.append(("Password Field", "✓ Not found", "pass"))

    return checks


def get_threat_indicators(url: str, feats: dict) -> list:
    threats = []

    brand_keywords = ["paypa1", "g00gle", "amaz0n", "micosoft", "faceb00k"]
    for kw in brand_keywords:
        if kw in url.lower():
            original = kw.replace("1", "l").replace("0", "o")
            threats.append((f"URL contains spoofed character '{kw}' (imitating {original})", "TYPOSQUATTING"))

    risky_tlds = [".tk", ".ml", ".ga", ".cf", ".gq"]
    for tld in risky_tlds:
        if url.lower().endswith(tld) or (tld + "/") in url.lower():
            threats.append((f"Uses high-risk free TLD {tld}, commonly seen in phishing attacks", "RISKY TLD"))
            break

    if feats.get("NoOfSubDomain", 0) >= 3:
        threats.append(("Multiple subdomains detected, often used to disguise legitimate domains", "SUSPICIOUS STRUCTURE"))

    if feats.get("HasPasswordField", 0) == 1:
        threats.append(("Page contains a password input field, possibly used to steal credentials", "CREDENTIAL HARVEST"))

    danger_paths = ["login", "verify", "update", "secure", "account", "signin"]
    for kw in danger_paths:
        if kw in url.lower():
            threats.append((f"URL path contains keyword '{kw}', commonly used in social engineering", "SOCIAL ENGINEERING"))
            break

    if not threats:
        threats.append(("No obvious threat indicators found", ""))
    return threats


# ── sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Session Info")
    remaining = MAX_SCANS - st.session_state["scan_count"]
    st.markdown(f"Scans used: **{st.session_state['scan_count']} / {MAX_SCANS}**")
    st.progress(st.session_state["scan_count"] / MAX_SCANS)
    if remaining <= 3:
        st.warning(f"⚠ Only {remaining} scan(s) left this session.")
    st.markdown("---")
    st.markdown("**Limitations**")
    st.markdown("""
- Max **15 scans** per session
- URLs must use `http://` or `https://`
- Private IPs & localhost blocked
- Max URL length: 2000 chars
- Batch scan: max **50 URLs** per file
    """)
    st.markdown("---")
    st.caption("CKJ Capstone · RF + DT + ANN · 51 features")


# ── header ──
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:1.5rem;">
    <div style="font-size:32px;">🛡️</div>
    <div>
        <div style="font-size:18px; font-weight:600; color:#e6edf3; letter-spacing:0.04em;">
            PHISHGUARD // Phishing URL Detection System
        </div>
        <div style="font-size:11px; color:#30a14e; letter-spacing:0.08em; margin-top:2px;">
            ▶ THREAT INTELLIGENCE · RF + DT + ANN · 51 FEATURES
        </div>
    </div>
    <div style="margin-left:auto; display:flex; gap:6px;">
        <span style="padding:2px 10px; border-radius:20px; font-size:10px;
              background:#0f2a0f; border:0.5px solid #30a14e; color:#30a14e;">
            ● ONLINE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Single Scan", "📂 Batch Scan", "📊 Model Comparison"])


# ── Tab 1: Single URL scan ──
with tab1:

    st.markdown("""
    <div class="terminal-card">
        <div class="terminal-topbar">
            <span class="dot-red"></span>
            <span class="dot-yellow"></span>
            <span class="dot-green"></span>
            <span class="term-title">SCANNER TERMINAL</span>
        </div>
    """, unsafe_allow_html=True)

    if "url_prefill" not in st.session_state:
        st.session_state["url_prefill"] = ""

    st.markdown("<div style='font-size:11px; color:#8b949e; margin-top:4px;'>Quick Examples:</div>",
                unsafe_allow_html=True)

    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    examples = {
        "google.com": "https://google.com",
        "github.com": "https://github.com",
        "paypa1.tk":  "http://paypa1-secure.login.tk/update",
        "192.168.1":  "http://192.168.1.1/login",
    }
    for col, (label, ex_url) in zip([ex_col1, ex_col2, ex_col3, ex_col4], examples.items()):
        with col:
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state["url_prefill"] = ex_url
                st.rerun()

    col_prompt, col_input = st.columns([1, 8])
    with col_prompt:
        st.markdown("<div style='color:#30a14e; padding-top:8px; font-size:14px;'>$&nbsp;scan</div>",
                    unsafe_allow_html=True)
    with col_input:
        url_input = st.text_input(
            label="URL",
            label_visibility="collapsed",
            placeholder="Enter a URL, e.g. http://paypa1-secure.tk/login",
            value=st.session_state["url_prefill"],
        )

    st.markdown("</div>", unsafe_allow_html=True)

    scan_btn = st.button("🔍 SCAN — Start Detection", use_container_width=True)

    if scan_btn and url_input.strip():
        url = url_input.strip()

        # check scan limit first
        if st.session_state["scan_count"] >= MAX_SCANS:
            st.error(f"❌ Scan limit reached ({MAX_SCANS} scans per session). Please reload the page to reset.")
        else:
            # validate the URL before sending it to the model
            is_valid, err_msg = validate_url(url)
            if not is_valid:
                st.warning(f"⚠ {err_msg}")
            else:
                st.session_state["scan_count"] += 1
                st.query_params["sc"] = str(st.session_state["scan_count"])
                if "sc_ts" not in st.query_params:
                    import time
                    st.query_params["sc_ts"] = str(int(time.time()))

                with st.spinner("[ ... ] Extracting features + running model predictions..."):
                    rf_prob, dt_prob, ann_prob, verdict, feats = predict(url)

                if verdict == "ERROR" or feats is None:
                    st.error("❌ Unable to access this URL. Please check if the URL is correct or the site is reachable.")
                else:
                    is_phishing = verdict == "PHISHING"
                    color        = "#f85149" if is_phishing else "#30a14e"
                    result_class = "result-danger" if is_phishing else "result-safe"
                    verdict_cn   = "⛔ Phishing URL" if is_phishing else "✅ Legitimate URL"
                    avg_prob     = (rf_prob + dt_prob + ann_prob) / 3

                    st.markdown(f"""
                    <div class="{result_class}">
                        <div style="display:flex; align-items:center; justify-content:space-between;">
                            <div>
                                <div style="font-size:20px; font-weight:600; color:{color};">
                                    {verdict_cn}
                                </div>
                                <div style="font-size:12px; color:#8b949e; margin-top:4px; word-break:break-all;">
                                    {url}
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:11px; color:#8b949e;">Avg. Confidence</div>
                                <div style="font-size:28px; font-weight:600; color:{color};">
                                    {avg_prob*100:.1f}%
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div style='font-size:11px; color:#8b949e; margin:12px 0 6px; letter-spacing:.06em;'>ML MODEL CONSENSUS</div>",
                                unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    for col, name, prob in [(m1, "Random Forest", rf_prob),
                                             (m2, "Decision Tree", dt_prob),
                                             (m3, "ANN", ann_prob)]:
                        with col:
                            m_verdict = "PHISHING" if prob >= 0.5 else "SAFE"
                            m_color   = "#f85149" if prob >= 0.5 else "#30a14e"
                            st.markdown(f"""
                            <div style="background:#161b22; border:0.5px solid #21262d;
                                        border-radius:8px; padding:10px; text-align:center;">
                                <div style="font-size:10px; color:#8b949e; letter-spacing:.05em; margin-bottom:6px;">{name}</div>
                                <div style="font-size:13px; font-weight:600; color:{m_color}; margin-bottom:6px;">{m_verdict}</div>
                                <div style="background:#21262d; height:6px; border-radius:3px; overflow:hidden;">
                                    <div style="background:{m_color}; height:100%; width:{prob*100:.1f}%; border-radius:3px;"></div>
                                </div>
                                <div style="font-size:11px; color:#8b949e; margin-top:4px;">{prob*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    left_col, right_col = st.columns(2)

                    with left_col:
                        checks = get_security_checks(url, feats)
                        st.markdown("""
                        <div class="terminal-card">
                            <div style="font-size:10px; color:#8b949e; letter-spacing:.08em; margin-bottom:10px;">
                                SECURITY CHECKS
                            </div>
                        """, unsafe_allow_html=True)
                        for check_name, check_result, status in checks:
                            css_class = f"check-{status}"
                            st.markdown(f"""
                            <div class="check-item">
                                <span class="check-label">{check_name}</span>
                                <span class="{css_class}">{check_result}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with right_col:
                        key_features = [
                            ("IsHTTPS",          feats.get("IsHTTPS", 0),          lambda v: "ok" if v == 1 else "bad",  lambda v: "✓ YES" if v == 1 else "✗ NO"),
                            ("URLLength",        feats.get("URLLength", 0),         lambda v: "ok" if v <= 50 else "bad",  lambda v: str(v)),
                            ("NoOfSubDomain",    feats.get("NoOfSubDomain", 0),     lambda v: "ok" if v <= 1 else "bad",  lambda v: str(v)),
                            ("IsDomainIP",       feats.get("IsDomainIP", 0),        lambda v: "ok" if v == 0 else "bad",  lambda v: "NO" if v == 0 else "YES"),
                            ("HasPasswordField", feats.get("HasPasswordField", 0),  lambda v: "ok" if v == 0 else "bad",  lambda v: "NO" if v == 0 else "YES"),
                            ("HasObfuscation",   feats.get("HasObfuscation", 0),    lambda v: "ok" if v == 0 else "bad",  lambda v: "NO" if v == 0 else "YES"),
                            ("HasTitle",         feats.get("HasTitle", 0),          lambda v: "ok" if v == 1 else "bad",  lambda v: "YES" if v == 1 else "NO"),
                            ("HasFavicon",       feats.get("HasFavicon", 0),        lambda v: "ok" if v == 1 else "bad",  lambda v: "YES" if v == 1 else "NO"),
                        ]
                        st.markdown("""
                        <div class="terminal-card">
                            <div style="font-size:10px; color:#8b949e; letter-spacing:.08em; margin-bottom:10px;">
                                KEY FEATURES （51 TOTAL）
                            </div>
                        """, unsafe_allow_html=True)
                        for feat_name, val, get_class, get_display in key_features:
                            css_class   = f"feat-{get_class(val)}"
                            display_val = get_display(val)
                            st.markdown(f"""
                            <div class="feature-row">
                                <span class="feat-key">{feat_name}</span>
                                <span class="{css_class}">{display_val}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    if is_phishing:
                        indicators = get_threat_indicators(url, feats)
                        if indicators and indicators[0][0] != "No obvious threat indicators found":
                            st.markdown("""
                            <div style="background:#160b0b; border:0.5px solid #f85149;
                                        border-left:3px solid #f85149; border-radius:8px;
                                        padding:12px 16px; margin-top:4px;">
                                <div style="font-size:10px; color:#f85149; letter-spacing:.08em; margin-bottom:8px;">
                                    ⚠ THREAT INDICATORS DETECTED
                                </div>
                            """, unsafe_allow_html=True)
                            for desc, tag in indicators:
                                tag_html = f'<span class="threat-tag">{tag}</span>' if tag else ""
                                st.markdown(f"""
                                <div style="display:flex; gap:8px; margin-bottom:4px; align-items:flex-start;">
                                    <span style="color:#f85149; margin-top:2px; flex-shrink:0;">▸</span>
                                    <span style="font-size:12px; color:#c9d1d9; line-height:1.5;">
                                        {desc}{tag_html}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("📋 View All 51 Features"):
                        feat_df = pd.DataFrame(
                            list(feats.items())[:51],
                            columns=["Feature", "Value"]
                        )
                        st.dataframe(feat_df, use_container_width=True, hide_index=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    report_lines = [
                        "=== PHISHGUARD DETECTION REPORT ===",
                        f"URL: {url}",
                        f"Result: {verdict_cn}",
                        f"Random Forest Confidence: {rf_prob*100:.1f}%",
                        f"Decision Tree Confidence:  {dt_prob*100:.1f}%",
                        f"ANN Confidence:            {ann_prob*100:.1f}%",
                        "",
                        "=== FEATURE VALUES ===",
                    ] + [f"{k}: {v}" for k, v in feats.items()]

                    st.download_button(
                        label="⬇ Download Report (.txt)",
                        data="\n".join(report_lines),
                        file_name="phishguard_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

    elif scan_btn:
        st.warning("⚠ Please enter a URL before scanning.")


# ── Tab 2: Batch Scan ──
with tab2:
    st.markdown("""
    <div class="terminal-card">
        <div class="terminal-topbar">
            <span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span>
            <span class="term-title">BATCH SCANNER</span>
        </div>
        <div style="font-size:12px; color:#8b949e; margin-top:4px;">
            Upload a CSV file with a column named <code>url</code>. Max 50 URLs per file.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df_in = pd.read_csv(uploaded)

        if "url" not in df_in.columns:
            st.error("❌ Column 'url' not found in CSV. Please check the column name.")
        elif len(df_in) > 50:
            st.error(f"❌ File contains {len(df_in)} URLs. Batch scan is limited to 50 URLs per file.")
        else:
            st.info(f"Found {len(df_in)} URLs. Starting scan...")
            results = []
            progress_bar = st.progress(0)

            for i, row_url in enumerate(df_in["url"]):
                row_url_str = str(row_url).strip()

                # validate each URL before scanning
                is_valid, err_msg = validate_url(row_url_str)
                if not is_valid:
                    results.append({
                        "URL":       row_url_str,
                        "Result":    f"Skipped: {err_msg}",
                        "RF Conf.":  "N/A",
                        "ANN Conf.": "N/A",
                    })
                else:
                    rf_p, dt_p, ann_p, verd, _ = predict(row_url_str)
                    if verd == "ERROR":
                        results.append({
                            "URL":       row_url_str,
                            "Result":    "Cannot Access",
                            "RF Conf.":  "N/A",
                            "ANN Conf.": "N/A",
                        })
                    else:
                        is_phish = (verd == "PHISHING")
                        # show confidence in the verdict, not raw phishing probability
                        # e.g. DT=100% phishing → shows 100% if result=Phishing, or 0% if result=Legit
                        def conf(p, phishing_result):
                            return f"{p*100:.1f}%" if phishing_result else f"{(1-p)*100:.1f}%"

                        results.append({
                            "URL":       row_url_str,
                            "Result":    "Phishing" if is_phish else "Legitimate",
                            "RF Conf.":  conf(rf_p, is_phish),
                            "ANN Conf.": conf(ann_p, is_phish),
                        })
                progress_bar.progress((i + 1) / len(df_in))

            df_out = pd.DataFrame(results)
            st.dataframe(df_out, use_container_width=True)

            phish_count = sum(1 for r in results if r["Result"] == "Phishing")
            safe_count  = sum(1 for r in results if r["Result"] == "Legitimate")
            skip_count  = len(results) - phish_count - safe_count
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Phishing", phish_count)
            c2.metric("🟢 Legitimate", safe_count)
            c3.metric("⚪ Skipped/Error", skip_count)

            csv_out = df_out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ Download Batch Results (.csv)",
                data=csv_out,
                file_name="batch_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# ── Tab 3: Model Performance Comparison ──
with tab3:
    st.markdown("""
    <div class="terminal-card">
        <div class="terminal-topbar">
            <span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span>
            <span class="term-title">MODEL PERFORMANCE COMPARISON</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # these numbers come from Step 6 / Step 8 / Step 10 in the notebook
    comparison_data = {
        "Model":            ["Random Forest", "Decision Tree", "ANN (Deep Learning)"],
        "Accuracy":         ["100.00%", "100.00%", "99.99%"],
        "Precision":        ["100.00%", "100.00%", "99.99%"],
        "Recall":           ["100.00%", "100.00%", "100.00%"],
        "F1 Score":         ["100.00%", "100.00%", "99.99%"],
        "Training Samples": ["188,636", "188,636", "188,636"],
        "Test Samples":     ["47,159",  "47,159",  "47,159"],
    }
    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="font-size:11px; color:#8b949e; margin-top:8px;">
        ※ 80% training / 20% test split on 235,795 URLs total
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px; color:#8b949e; letter-spacing:.08em; margin-bottom:8px;">
        TOP 10 FEATURE IMPORTANCE（Random Forest）
    </div>
    """, unsafe_allow_html=True)

    # feature_importances_ is built into RandomForest after training
    importances = rf_model.feature_importances_
    feature_names = [
        "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex",
        "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength",
        "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
        "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
        "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
        "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
        "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
        "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
        "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
        "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
        "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS",
        "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef", "TLD_encoded",
    ]

    n_features = min(len(importances), len(feature_names))
    imp_df = pd.DataFrame({
        "Feature":    feature_names[:n_features],
        "Importance": importances[:n_features]
    }).sort_values("Importance", ascending=False).head(10)

    st.bar_chart(imp_df.set_index("Feature")["Importance"])


# ── footer ──
st.markdown("""
<hr style="border-color:#21262d; margin-top:2rem;">
<div style="text-align:center; font-size:11px; color:#8b949e; padding-bottom:1rem;">
    PHISHGUARD · CKJ Capstone Project · Random Forest + Decision Tree + ANN · 51 Features
</div>
""", unsafe_allow_html=True)
