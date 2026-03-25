import streamlit as st

from src.config import METRICS_PATH
from src.predict import PredictionService
from src.utils import load_json


st.set_page_config(page_title="Phishing URL Detector", layout="wide")


@st.cache_resource
def get_prediction_service():
    return PredictionService()


@st.cache_data
def get_metrics():
    if METRICS_PATH.exists():
        return load_json(METRICS_PATH)
    return None


def main():
    st.title("Hybrid CNN Phishing URL Detector")
    st.write(
        "This app combines a character-level CNN with handcrafted URL risk signals "
        "to classify whether a URL is likely phishing or legitimate."
    )

    with st.sidebar:
        st.header("Project")
        st.write("Model: Hybrid CNN + handcrafted URL features")
        st.write("Frameworks: PyTorch, scikit-learn, Streamlit")
        metrics = get_metrics()
        if metrics:
            test_metrics = metrics.get("test_metrics", {})
            st.metric("Test F1", f"{test_metrics.get('f1', 0):.3f}")
            st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
        else:
            st.info("Train the model first to display saved evaluation metrics.")

    sample_urls = [
        "https://accounts.google.com",
        "http://secure-paypal-login.verify-user-account.ru/login",
        "http://192.168.0.1/free-gift-card?claim=now",
    ]

    url = st.text_input("Enter a URL", value=sample_urls[1], placeholder="https://example.com")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Analyze URL", use_container_width=True):
            try:
                service = get_prediction_service()
                result = service.predict(url)
            except FileNotFoundError:
                st.error(
                    "Model artifacts are missing. Run the training pipeline first: "
                    "`python -m src.train --data-path <dataset>`"
                )
                return

            st.subheader("Prediction")
            if result.label == "Phishing":
                st.error(f"{result.label} ({result.risk_level} risk)")
            else:
                st.success(f"{result.label} ({result.risk_level} risk)")

            st.metric("Confidence", f"{result.confidence:.2%}")
            st.metric("Phishing probability", f"{result.probability:.2%}")

    with col2:
        st.subheader("Example URLs")
        for sample in sample_urls:
            st.code(sample)

    st.subheader("Handcrafted Signals")
    st.caption("These features are concatenated with the CNN representation before final classification.")
    if "result" in locals():
        feature_items = list(result.handcrafted_features.items())
        left, right = st.columns(2)
        midpoint = len(feature_items) // 2
        for name, value in feature_items[:midpoint]:
            left.write(f"**{name}**: {value:.3f}")
        for name, value in feature_items[midpoint:]:
            right.write(f"**{name}**: {value:.3f}")
    else:
        st.info("Run an analysis to inspect the URL features used by the hybrid model.")

    st.subheader("How It Works")
    st.markdown(
        """
        - The CNN reads the URL as a sequence of characters.
        - Handcrafted features capture explicit risk patterns such as IP-based domains and suspicious symbols.
        - Both signals are fused in one classifier for final phishing scoring.
        """
    )


if __name__ == "__main__":
    main()
