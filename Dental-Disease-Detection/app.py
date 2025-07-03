import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import tempfile
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO


# Model path        
MODEL_PATH = r"D:\mini project\Dental-Disease-Detection\weights\best.pt"


# Class names from your project
CLASS_NAMES = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']

# Page setup
st.set_page_config(
    page_title="Dental Health AI Assistant",
    page_icon="🦷",
    layout="wide"
)

# Function to set background image
def add_bg_from_local(image_path):
    try:
        if not os.path.exists(image_path):
            st.error(f"Background image not found at: {image_path}")
            return

        with open(image_path, "rb") as file:
            bg_image = file.read()
        bg_image_base64 = base64.b64encode(bg_image).decode()

        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error setting background image: {e}")

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    return YOLO(model_path)

def get_disease_suggestion(class_name):
    suggestions = {
        'Caries': [
            "Brush with fluoride toothpaste.",
            "Limit sugary foods and drinks.",
            "Schedule a dental check-up.",
            "Consider dental fillings if necessary."
        ],
        'Ulcer': [
            "Use topical oral gels for relief.",
            "Avoid spicy or acidic foods.",
            "Maintain proper oral hygiene.",
            "Monitor healing progress."
        ],
        'Tooth Discoloration': [
            "Try whitening toothpaste.",
            "Avoid staining beverages (coffee, tea).",
            "Consult a dentist for professional cleaning.",
            "Consider teeth whitening treatments."
        ],
        'Gingivitis': [
            "Floss daily to remove plaque.",
            "Use an antimicrobial mouthwash.",
            "Brush gently along the gum line.",
            "Schedule a professional cleaning."
        ]
    }
    return suggestions.get(class_name, ["Consult your dentist for professional advice."])

def generate_pdf_report(detections, original_image_path, analyzed_image_path):
    """Generates a detailed PDF report."""
    pdf_buffer = BytesIO()

    # Create a canvas object
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Dental Health Analysis Report")

    # Original Image
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "Original Image:")
    c.drawImage(original_image_path, 50, height - 300, width=200, height=200)

    # Analyzed Image
    c.drawString(300, height - 100, "Analyzed Image:")
    c.drawImage(analyzed_image_path, 300, height - 300, width=200, height=200)

    # Analysis Results
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 350, "AI Analysis Results:")

    if not detections:
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 380, "No dental issues detected. Maintain good oral hygiene!")
    else:
        y_position = height - 380
        for class_name, data in detections.items():
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, f"{class_name} (Max Confidence: {data['max_confidence']:.2%})")
            y_position -= 20
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, f"Detected: {data['count']} time(s)")
            y_position -= 20
            c.drawString(50, y_position, "Suggestions:")
            y_position -= 20
            for suggestion in data['suggestions']:
                c.drawString(70, y_position, f"- {suggestion}")
                y_position -= 20
                if y_position < 100:  # Start a new page if space is insufficient
                    c.showPage()
                    y_position = height - 50

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer   
# CSS Styling for Center Alignment
def center_align_content():
    st.markdown(
        """
        <style>
        .center-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .stMarkdown h3 {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.title("🦷 AI Dental Health Assistant")
    center_align_content()  
    st.markdown("""
    **Detect dental conditions and get personalized recommendations**
    Upload a clear intraoral photo or use your camera for analysis and preventive care suggestions.
    """)

    # Sidebar
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload a clear dental photo or use the camera.
        2. Wait for AI analysis.
        3. Review results & suggestions.
        4. Download the report.
        """)
        st.divider()
        st.markdown("**Disclaimer**")
        st.markdown("This AI assistant provides preliminary suggestions. Always consult a licensed dentist for professional diagnosis.")
    
    
    # File uploader and camera input
    st.subheader("Choose an input method:")
    input_method = st.radio(
        "Select input type:",
        options=["Upload Image", "Use Camera"],
        horizontal=True
    )

    uploaded_file = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Dental Photo", 
            type=['jpg', 'jpeg', 'png'],
            help="Max file size: 5MB"
        )
    elif input_method == "Use Camera":
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            uploaded_file = camera_file

    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name  # Path of the saved uploaded file

            image = Image.open(temp_file_path).convert("RGB")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
                
            with col2:
                with st.spinner("Analyzing dental health..."):
                    model = load_model(MODEL_PATH)
                    results = model(image)
                    res_plotted = results[0].plot()
                    
                    if res_plotted.shape[-1] == 3:
                        res_plotted = res_plotted[..., ::-1]
                    
                    st.image(res_plotted, caption="AI Analysis Results", use_container_width=True)
                    
                    detections = {}
                    if len(results[0].boxes) == 0:
                        st.success("🎉 No dental issues detected! Maintain good oral hygiene!")
                    else:
                        st.markdown('<div class="center-content">', unsafe_allow_html=True)  # Start center alignment
                        st.subheader("AI Analysis Results and Suggestions")
                        for box in results[0].boxes:
                            cls = int(box.cls)
                            conf = float(box.conf)
                            class_name = CLASS_NAMES[cls]
                            
                            if class_name not in detections:
                                detections[class_name] = {
                                    'max_confidence': conf,
                                    'count': 1,
                                    'suggestions': get_disease_suggestion(class_name)
                                }
                            else:
                                detections[class_name]['count'] += 1
                                if conf > detections[class_name]['max_confidence']:
                                    detections[class_name]['max_confidence'] = conf

                        for class_name, data in detections.items():
                            st.markdown(f"### {class_name} (Max Confidence: {data['max_confidence']:.2%})")
                            st.markdown(f"- **Detected**: {data['count']} time{'s' if data['count'] > 1 else ''}")
                            st.markdown("**Recommended Actions:**")
                            for suggestion in data['suggestions']:
                                st.markdown(f"  - {suggestion}")
                            st.divider()

                    # Save the analysis report as a PDF
                    if "pdf_ready" not in st.session_state:
                        st.session_state.pdf_ready = False

                    if st.button("Generate Report"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as analyzed_image:
                            analyzed_image_path = analyzed_image.name
                            Image.fromarray(res_plotted).save(analyzed_image_path)

                        pdf = generate_pdf_report(
                            detections=detections,
                            original_image_path=temp_file_path,
                            analyzed_image_path=analyzed_image_path
                        )
                        st.session_state.pdf_data = pdf
                        st.session_state.pdf_ready = True
                        st.success("Report generated! Download it below.")

                    if st.session_state.pdf_ready:
                        st.download_button(
                            label="Download Report",
                            data=st.session_state.pdf_data,
                            file_name="Dental_Health_Report.pdf",
                            mime="application/pdf"
                        )
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    bg_image_path = r"D:\mini project\Dental-Disease-Detection\assets\background.jpg"
    add_bg_from_local(bg_image_path)
    main()
