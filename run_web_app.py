from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import json
import secrets
from datetime import datetime
from forensic_fingerprint_matcher import ForensicFingerprintMatcher, MatchResult, logger
import io
import cv2
import numpy as np

# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import black, blue, red
from reportlab.lib.units import inch

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'forensic_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.secret_key = secrets.token_hex(16) # Set a strong, random secret key

# Ensure upload and results folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize matcher
forensic_config = {
    'sift_features': 4000,
    'lowe_ratio': 0.7,
    'min_match_count': 15,
    'geometric_verification': True,
    'adaptive_threshold': True,
    'min_quality_score': 0.2,
    'audit_logging': True
}
matcher = ForensicFingerprintMatcher(forensic_config)

# In-memory storage for case results (for demonstration)
case_results = {}

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        flash('Please select both fingerprint images.')
        return redirect(request.url)

    file1 = request.files['file1']
    file2 = request.files['file2']
    case_id = request.form.get('case_id')
    operator_id = request.form.get('operator_id')

    if file1.filename == '' or file2.filename == '' or not case_id or not operator_id:
        flash('All fields (Case ID, Operator ID, and both files) are required.')
        return redirect(request.url)

    try:
        # Save files to disk
        filename1 = f"{case_id}_img1_{file1.filename}"
        filename2 = f"{case_id}_img2_{file2.filename}"
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        file1.save(filepath1)
        file2.save(filepath2)

        # Perform matching using the forensic matcher module
        result = matcher.match_fingerprints(filepath1, filepath2, case_id, operator_id)

        # Store result for display and PDF generation
        case_results[case_id] = {
            'result': result,
            'img1_path': filepath1,
            'img2_path': filepath2
        }

        # Save the structured JSON report
        result_filename_json = f"{case_id}_result.json"
        result_filepath_json = os.path.join(app.config['RESULTS_FOLDER'], result_filename_json)
        with open(result_filepath_json, 'w') as f:
            json.dump(result.to_serializable_dict(), f, indent=2)

        flash("Analysis complete. Check the report below.")
        return redirect(url_for('display_results', case_id=case_id))

    except Exception as e:
        flash(f"An error occurred during analysis: {e}")
        return redirect(url_for('index'))

@app.route('/results/<case_id>')
def display_results(case_id):
    if case_id not in case_results:
        flash("Case results not found.")
        return redirect(url_for('index'))

    data = case_results[case_id]
    result_data = data['result']
    result_filename_json = f"{case_id}_result.json"
    
    return render_template('result.html', result=result_data, result_file=result_filename_json)

@app.route('/download-json/<filename>')
def serve_result_file(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename), as_attachment=True)


# PDF GENERATION ROUTE
@app.route('/download-pdf/<case_id>')
def download_pdf_report(case_id):
    if case_id not in case_results:
        flash("Case results not found for PDF generation.")
        return redirect(url_for('index'))

    data = case_results[case_id]
    result = data['result']
    img1_path = data['img1_path']
    img2_path = data['img2_path']
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    # Renamed 'Heading2' to 'ReportHeading' to avoid the KeyError
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=24, leading=28, alignment=TA_CENTER,
                              spaceAfter=20, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubtitleStyle', fontSize=14, leading=16, alignment=TA_CENTER,
                              spaceAfter=10, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ReportHeading', fontSize=14, leading=16, fontName='Helvetica-Bold',
                              spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='BodyTextLeft', fontSize=10, leading=12, alignment=TA_LEFT,
                              spaceAfter=4))
    styles.add(ParagraphStyle(name='BodyTextCenter', fontSize=10, leading=12, alignment=TA_CENTER,
                              spaceAfter=4))
    styles.add(ParagraphStyle(name='OfficialSeal', fontSize=8, leading=10, alignment=TA_CENTER,
                              spaceBefore=20, spaceAfter=2))
    styles.add(ParagraphStyle(name='MatchStatusGreen', fontSize=16, leading=18, alignment=TA_CENTER,
                              fontName='Helvetica-Bold', textColor=blue, spaceAfter=10))
    styles.add(ParagraphStyle(name='MatchStatusRed', fontSize=16, leading=18, alignment=TA_CENTER,
                              fontName='Helvetica-Bold', textColor=red, spaceAfter=10))


    Story = []

    # Government Header/Seal (example)
    Story.append(Paragraph("GOVERNMENT OF INDIA", styles['SubtitleStyle']))
    Story.append(Paragraph("MINISTRY OF HOME AFFAIRS", styles['SubtitleStyle']))
    Story.append(Paragraph("NATIONAL FORENSIC SCIENCES UNIVERSITY", styles['SubtitleStyle']))
    Story.append(Spacer(0, 0.2 * inch))
    Story.append(Paragraph("FORENSIC FINGERPRINT MATCHING REPORT", styles['TitleStyle']))
    Story.append(Spacer(0, 0.3 * inch))

    # Case Details
    # Use the new style name 'ReportHeading' here
    Story.append(Paragraph(f"Case ID: <font color='blue'>{result.case_id}</font>", styles['ReportHeading']))
    Story.append(Paragraph(f"Operator ID: {result.operator_id}", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Algorithm Version: {result.algorithm_version}", styles['BodyTextLeft']))
    Story.append(Spacer(0, 0.2 * inch))

    # Match Status
    if result.is_match:
        Story.append(Paragraph("✅ MATCH FOUND", styles['MatchStatusGreen']))
    else:
        Story.append(Paragraph("❌ NO MATCH", styles['MatchStatusRed']))
    Story.append(Spacer(0, 0.1 * inch))
    Story.append(Paragraph(f"<b>Confidence Score:</b> {result.confidence_score * 100:.2f}%", styles['BodyTextCenter']))
    Story.append(Spacer(0, 0.2 * inch))

    # Summary Table
    data_table = [
        ['Metric', 'Value'],
        ['Match Count', result.match_count],
        ['Threshold Used', f"{result.threshold_used:.2f}"],
        ['Geometric Verification', 'Valid' if result.geometric_verification else 'Invalid'],
        ['Image Quality (Print 1)', f"{result.quality_score_img1:.2f}"],
        ['Image Quality (Print 2)', f"{result.quality_score_img2:.2f}"],
        ['Processing Time', f"{result.processing_time:.2f} seconds"],
        ['Image 1 Hash (SHA256)', result.image1_hash],
        ['Image 2 Hash (SHA256)', result.image2_hash],
    ]
    table = Table(data_table, colWidths=[2.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
        ('GRID', (0, 0), (-1, -1), 1, black),
        ('FONTSIZE', (0,0), (-1,-1), 10)
    ]))
    Story.append(table)
    Story.append(Spacer(0, 0.5 * inch))

    # Fingerprint Images with Mapped Matches
    Story.append(Paragraph("Visual Match Overview", styles['ReportHeading']))
    Story.append(Spacer(0, 0.2 * inch))

    try:
        # Re-detect keypoints and descriptors to get them for drawing
        img1_orig = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_orig = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1_orig is None or img2_orig is None:
            raise ValueError("Original images not found for drawing matches.")
        
        # Enhance images (must be consistent with matching process)
        enhanced1 = matcher._enhance_fingerprint(img1_orig)
        enhanced2 = matcher._enhance_fingerprint(img2_orig)

        kp1, des1 = matcher.sift.detectAndCompute(enhanced1, None)
        kp2, des2 = matcher.sift.detectAndCompute(enhanced2, None)

        # Re-match
        matches = matcher.flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < matcher.config['lowe_ratio'] * n.distance:
                    good_matches.append(m)
        
        inlier_mask = None
        if result.geometric_verification and len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if M is not None:
                inlier_mask = mask.ravel().tolist()

        img_matches = cv2.drawMatches(
            cv2.cvtColor(img1_orig, cv2.COLOR_GRAY2BGR), kp1, 
            cv2.cvtColor(img2_orig, cv2.COLOR_GRAY2BGR), kp2, 
            good_matches, None, 
            matchColor=(0, 255, 0) if result.is_match else (0, 0, 255),
            singlePointColor=(255, 0, 0),
            matchesMask=inlier_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        _, img_encoded = cv2.imencode('.png', img_matches)
        img_buffer = io.BytesIO(img_encoded.tobytes())
        pil_image = Image(img_buffer)
        
        img_width, img_height = pil_image.drawWidth, pil_image.drawHeight
        max_width = doc.width
        scale_factor = max_width / img_width
        
        pil_image.drawWidth = max_width
        pil_image.drawHeight = img_height * scale_factor

        Story.append(pil_image)
        Story.append(Paragraph("<i>Matched Keypoints (Green: Inlier, Red: No Match/Outlier)</i>", styles['BodyTextCenter']))
        Story.append(Spacer(0, 0.2 * inch))

    except Exception as e:
        logger.error(f"Error drawing matches for PDF: {e}")
        Story.append(Paragraph(f"<i>Could not generate visual match overview due to an error: {e}</i>", styles['BodyTextLeft']))
        Story.append(Spacer(0, 0.2 * inch))

    Story.append(PageBreak())

    Story.append(Paragraph("<b>Disclaimer:</b> This report is generated by an automated forensic fingerprint matching system. The results are based on advanced algorithms and image processing techniques. Human expert verification is recommended for critical decisions.", styles['BodyTextLeft']))
    Story.append(Spacer(0, 0.1 * inch))
    Story.append(Paragraph(f"Location of Analysis: Alappuzha, Kerala, India", styles['BodyTextLeft']))
    Story.append(Spacer(0, 0.5 * inch))
    Story.append(Paragraph("_____________________________", styles['BodyTextCenter']))
    Story.append(Paragraph("Official Signature / Digital Seal", styles['BodyTextCenter']))
    Story.append(Spacer(0, 0.1 * inch))
    Story.append(Paragraph("<i>(This document is digitally signed and considered valid without physical signature)</i>", styles['BodyTextCenter']))


    doc.build(Story)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f"ForensicReport_{case_id}.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)