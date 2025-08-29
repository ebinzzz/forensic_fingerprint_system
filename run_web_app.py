from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import json
import secrets
import copy
from datetime import datetime
from forensic_fingerprint_matcher import ForensicFingerprintMatcher, MatchResult, logger
import io
import cv2
import numpy as np
import glob
from pathlib import Path

# For PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import black, blue, red, green
from reportlab.lib.units import inch

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE_FOLDER'] = 'database'  # Folder containing database fingerprints
app.config['RESULTS_FOLDER'] = 'forensic_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.secret_key = secrets.token_hex(16)

# Ensure all folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)
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

# In-memory storage for case results (prevent duplicates)
case_results = {}

def get_database_images():
    """Get all image files from the database folder."""
    supported_extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.pgm', '.ppm']
    database_images = []
    
    for ext in supported_extensions:
        pattern = os.path.join(app.config['DATABASE_FOLDER'], f"*{ext}")
        database_images.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern = os.path.join(app.config['DATABASE_FOLDER'], f"*{ext.upper()}")
        database_images.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    return sorted(list(set(database_images)))

def debug_result_consistency(case_id, results):
    """Enhanced debugging function to identify result inconsistencies."""
    logger.info("=== ENHANCED RESULT CONSISTENCY CHECK ===")
    logger.info(f"Case ID: {case_id}")
    logger.info(f"Total results: {len(results)}")
    
    inconsistencies_found = 0
    
    for i, r in enumerate(results):
        result = r['result']
        database_file = r['database_file']
        
        logger.info(f"\n--- Result #{i+1}: {database_file} ---")
        logger.info(f"is_match: {result.is_match} (type: {type(result.is_match)}, repr: {repr(result.is_match)})")
        logger.info(f"confidence_score: {result.confidence_score} (type: {type(result.confidence_score)})")
        logger.info(f"match_count: {result.match_count} (type: {type(result.match_count)})")
        
        # Check for type issues
        if not isinstance(result.is_match, bool):
            logger.error(f"ERROR: is_match is not boolean! Type: {type(result.is_match)}, Value: {result.is_match}")
            inconsistencies_found += 1
        
        # Check for logical inconsistencies
        if result.is_match and result.confidence_score < 0.2:
            logger.error(f"INCONSISTENCY: Match=True but confidence={result.confidence_score}")
            inconsistencies_found += 1
        
        if not result.is_match and result.confidence_score > 0.8:
            logger.error(f"INCONSISTENCY: Match=False but confidence={result.confidence_score}")
            inconsistencies_found += 1
        
        if result.is_match and result.match_count < 10:
            logger.error(f"INCONSISTENCY: Match=True but match_count={result.match_count}")
            inconsistencies_found += 1
        
        # Test boolean operations
        try:
            bool_test = bool(result.is_match)
            equality_test = result.is_match == True
            identity_test = result.is_match is True
            
            logger.info(f"Boolean tests - bool(): {bool_test}, ==True: {equality_test}, is True: {identity_test}")
            
            if bool_test != equality_test or bool_test != identity_test:
                logger.error(f"BOOLEAN INCONSISTENCY detected for {database_file}")
                inconsistencies_found += 1
        except Exception as bool_error:
            logger.error(f"Boolean test failed for {database_file}: {bool_error}")
            inconsistencies_found += 1
    
    logger.info(f"\n=== CONSISTENCY CHECK COMPLETE ===")
    logger.info(f"Total inconsistencies found: {inconsistencies_found}")
    return inconsistencies_found == 0

def validate_match_result_enhanced(result, database_file):
    """Enhanced validation with detailed type and value checking."""
    try:
        logger.info(f"=== ENHANCED VALIDATION: {database_file} ===")
        
        # Basic attribute checks
        is_match = result.is_match
        confidence = result.confidence_score
        match_count = result.match_count
        
        logger.info(f"is_match: {is_match} (type: {type(is_match)}, repr: {repr(is_match)})")
        logger.info(f"confidence: {confidence} (type: {type(confidence)})")
        logger.info(f"match_count: {match_count} (type: {type(match_count)})")
        
        # Type validation
        type_errors = []
        if not isinstance(is_match, bool):
            type_errors.append(f"is_match should be bool, got {type(is_match)}")
        if not isinstance(confidence, (int, float)):
            type_errors.append(f"confidence should be numeric, got {type(confidence)}")
        if not isinstance(match_count, int):
            type_errors.append(f"match_count should be int, got {type(match_count)}")
        
        if type_errors:
            logger.error(f"TYPE ERRORS for {database_file}: {type_errors}")
            return False
        
        # Value validation
        value_errors = []
        if confidence < 0 or confidence > 1:
            value_errors.append(f"confidence {confidence} out of range [0,1]")
        if match_count < 0:
            value_errors.append(f"match_count {match_count} cannot be negative")
        
        if value_errors:
            logger.error(f"VALUE ERRORS for {database_file}: {value_errors}")
            return False
        
        # Logical consistency
        logic_warnings = []
        if is_match and confidence < 0.2:
            logic_warnings.append(f"is_match=True but low confidence {confidence}")
        if not is_match and confidence > 0.8:
            logic_warnings.append(f"is_match=False but high confidence {confidence}")
        if is_match and match_count < 10:
            logic_warnings.append(f"is_match=True but low match_count {match_count}")
        
        if logic_warnings:
            logger.warning(f"LOGIC WARNINGS for {database_file}: {logic_warnings}")
        
        # Serialization test
        try:
            serialized = result.to_serializable_dict()
            serialized_is_match = serialized.get('is_match')
            if serialized_is_match != is_match:
                logger.error(f"SERIALIZATION MISMATCH: original={is_match}, serialized={serialized_is_match}")
                return False
        except Exception as ser_error:
            logger.error(f"SERIALIZATION FAILED for {database_file}: {ser_error}")
            return False
        
        logger.info(f"VALIDATION PASSED for {database_file}")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced validation failed for {database_file}: {e}")
        return False

def validate_match_result(result, database_file):
    """Standard validation function - kept for compatibility."""
    return validate_match_result_enhanced(result, database_file)

def create_comparison_visualization(suspect_path, database_path, result):
    """Create a visualization showing the comparison between two fingerprints - FIXED VERSION."""
    try:
        # Load images
        img1 = cv2.imread(suspect_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(database_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            logger.error(f"Failed to load images for visualization: {suspect_path}, {database_path}")
            return None
        
        # Enhance images for visualization
        enhanced1 = matcher._enhance_fingerprint(img1)
        enhanced2 = matcher._enhance_fingerprint(img2)
        
        # Detect keypoints and descriptors
        kp1, des1 = matcher.sift.detectAndCompute(enhanced1, None)
        kp2, des2 = matcher.sift.detectAndCompute(enhanced2, None)
        
        if des1 is None or des2 is None:
            # Create side-by-side image without matches
            combined = np.hstack([cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), 
                                cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)])
            
            # Add "NO FEATURES DETECTED" text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "NO FEATURES DETECTED", (50, 50), 
                       font, 1.0, (0, 0, 255), 2)
            return combined
        
        # Find matches
        matches = matcher.flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < matcher.config['lowe_ratio'] * n.distance:
                    good_matches.append(m)
        
        # Perform geometric verification if enabled
        inlier_mask = None
        if result.geometric_verification and len(good_matches) >= 4:
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                if M is not None:
                    inlier_mask = mask.ravel().tolist()
            except Exception as geom_error:
                logger.warning(f"Geometric verification visualization failed: {geom_error}")
                inlier_mask = None
        
        # CRITICAL FIX: Use result.is_match directly with proper validation
        is_match = bool(result.is_match)  # Ensure boolean conversion for safety
        
        # Enhanced validation logging
        logger.info(f"VISUALIZATION - File: {os.path.basename(database_path)}")
        logger.info(f"VISUALIZATION - result.is_match: {result.is_match} (type: {type(result.is_match)})")
        logger.info(f"VISUALIZATION - converted is_match: {is_match}")
        logger.info(f"VISUALIZATION - result.confidence_score: {result.confidence_score}")
        logger.info(f"VISUALIZATION - result.match_count: {result.match_count}")
        
        # CRITICAL VALIDATION: Check for inconsistencies and force consistency
        if is_match and result.confidence_score < 0.2:
            logger.error(f"CRITICAL MISMATCH: is_match=True but confidence={result.confidence_score} for {database_path}")
            logger.warning(f"FORCING is_match=False due to low confidence")
            is_match = False
            
        if not is_match and result.confidence_score > 0.8:
            logger.error(f"CRITICAL MISMATCH: is_match=False but confidence={result.confidence_score} for {database_path}")
            logger.warning(f"FORCING is_match=True due to high confidence")
            is_match = True
        
        # Draw matches - Use the validated boolean value
        match_color = (0, 255, 0) if is_match else (0, 0, 255)
        img_matches = cv2.drawMatches(
            cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), kp1,
            cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), kp2,
            good_matches, None,
            matchColor=match_color,
            singlePointColor=(255, 0, 0),
            matchesMask=inlier_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Add text overlay with match information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # CRITICAL FIX: Use the validated boolean for display
        if is_match:
            status_text = "MATCH: YES"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "MATCH: NO" 
            status_color = (0, 0, 255)  # Red
            
        cv2.putText(img_matches, status_text, (10, 30), font, font_scale, status_color, thickness)
        
        # Confidence score
        conf_text = f"Confidence: {result.confidence_score:.3f}"
        cv2.putText(img_matches, conf_text, (10, 60), font, font_scale, (255, 255, 255), thickness)
        
        # Match count
        match_text = f"Matches: {result.match_count}"
        cv2.putText(img_matches, match_text, (10, 90), font, font_scale, (255, 255, 255), thickness)
        
        # Quality scores
        avg_quality = (result.quality_score_img1 + result.quality_score_img2) / 2
        quality_text = f"Quality: {avg_quality:.3f}"
        cv2.putText(img_matches, quality_text, (10, 120), font, font_scale, (255, 255, 255), thickness)
        
        # Final validation log
        logger.info(f"VISUALIZATION COMPLETE - {database_path}: Status='{status_text}', Color={'Green' if is_match else 'Red'}")
        
        return img_matches
        
    except Exception as e:
        logger.error(f"Error creating visualization for {database_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files:
        flash('Please select a suspect fingerprint image.')
        return redirect(request.url)

    file1 = request.files['file1']  # Suspect print
    file2 = request.files.get('file2')  # Database print (optional)
    case_id = request.form.get('case_id')
    operator_id = request.form.get('operator_id')

    if file1.filename == '' or not case_id or not operator_id:
        flash('Case ID, Operator ID, and suspect fingerprint are required.')
        return redirect(request.url)

    # PREVENT DUPLICATE PROCESSING
    if case_id in case_results:
        flash(f'Case {case_id} already processed. Results displayed below.')
        return redirect(url_for('display_results', case_id=case_id))

    try:
        # Save suspect fingerprint with unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename1 = f"{case_id}_{timestamp}_suspect_{file1.filename}"
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file1.save(filepath1)

        results = []
        visualizations = []
        processed_files = set()  # Track processed files to prevent duplicates

        if file2 and file2.filename != '':
            # Single comparison mode
            filename2 = f"{case_id}_{timestamp}_database_{file2.filename}"
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file2.save(filepath2)

            logger.info(f"Starting single comparison for case {case_id}")
            result = matcher.match_fingerprints(filepath1, filepath2, case_id, operator_id)
            
            # ENHANCED VALIDATION
            if not validate_match_result_enhanced(result, filename2):
                logger.error(f"Validation failed for {filename2}")
                flash("Validation error occurred during analysis.")
                return redirect(url_for('index'))
            
            # DEEP COPY to prevent reference issues
            result_copy = copy.deepcopy(result)
            
            results.append({
                'result': result_copy,
                'database_file': filename2,
                'database_path': filepath2
            })

            # Create visualization with validation
            logger.info(f"Creating visualization for {filename2} - Match: {result_copy.is_match}")
            visualization = create_comparison_visualization(filepath1, filepath2, result_copy)
            if visualization is not None:
                viz_filename = f"{case_id}_single_comparison.png"
                viz_path = os.path.join(app.config['RESULTS_FOLDER'], viz_filename)
                cv2.imwrite(viz_path, visualization)
                visualizations.append(viz_filename)
                logger.info(f"Visualization saved: {viz_path}")
            else:
                visualizations.append(None)
                logger.warning(f"Visualization creation failed for {filename2}")

        else:
            # Database scanning mode
            database_images = get_database_images()
            
            if not database_images:
                flash('No images found in the database folder.')
                return redirect(url_for('index'))

            logger.info(f'Starting database scan for case {case_id} against {len(database_images)} images')
            flash(f'Scanning against {len(database_images)} database images...')
            
            for i, db_image_path in enumerate(database_images):
                try:
                    db_filename = os.path.basename(db_image_path)
                    
                    # Skip if already processed (prevent duplicates)
                    if db_filename in processed_files:
                        logger.warning(f"Skipping duplicate: {db_filename}")
                        continue
                    
                    processed_files.add(db_filename)
                    sub_case_id = f"{case_id}_{i+1:03d}"
                    
                    logger.info(f"Processing {i+1}/{len(database_images)}: {db_filename}")
                    result = matcher.match_fingerprints(filepath1, db_image_path, sub_case_id, operator_id)
                    
                    # ENHANCED VALIDATION
                    if not validate_match_result_enhanced(result, db_filename):
                        logger.error(f"Validation failed for {db_filename} - skipping")
                        continue
                    
                    # DEEP COPY to prevent reference issues
                    result_copy = copy.deepcopy(result)
                    
                    # ADDITIONAL CONSISTENCY CHECK
                    logger.info(f"STORING RESULT - File: {db_filename}, is_match: {result_copy.is_match}, confidence: {result_copy.confidence_score}")
                    
                    results.append({
                        'result': result_copy,
                        'database_file': db_filename,
                        'database_path': db_image_path
                    })

                    # Create visualization with enhanced validation
                    logger.info(f"Creating visualization for {db_filename} - Match: {result_copy.is_match}")
                    visualization = create_comparison_visualization(filepath1, db_image_path, result_copy)
                    if visualization is not None:
                        viz_filename = f"{sub_case_id}_comparison.png"
                        viz_path = os.path.join(app.config['RESULTS_FOLDER'], viz_filename)
                        cv2.imwrite(viz_path, visualization)
                        visualizations.append(viz_filename)
                        logger.info(f"Visualization saved: {viz_path}")
                    else:
                        visualizations.append(None)
                        logger.warning(f"Visualization creation failed for {db_filename}")

                except Exception as e:
                    logger.error(f"Error processing database image {db_image_path}: {e}")
                    visualizations.append(None)
                    continue

        # FINAL CONSISTENCY CHECK
        if not debug_result_consistency(case_id, results):
            logger.error(f"CRITICAL: Consistency check failed for case {case_id}")
            flash("Warning: Result inconsistencies detected. Check logs.")

        # Store results - PREVENT DUPLICATE STORAGE
        case_results[case_id] = {
            'results': results,
            'suspect_path': filepath1,
            'visualizations': visualizations,
            'scan_mode': file2 is None or file2.filename == '',
            'timestamp': timestamp
        }

        # Save comprehensive JSON report
        result_filename_json = f"{case_id}_comprehensive_result.json"
        result_filepath_json = os.path.join(app.config['RESULTS_FOLDER'], result_filename_json)
        
        # Calculate summary stats
        matches_found = sum(1 for r in results if r['result'].is_match)
        
        json_data = {
            'case_id': case_id,
            'operator_id': operator_id,
            'timestamp': datetime.now().isoformat(),
            'scan_mode': case_results[case_id]['scan_mode'],
            'total_comparisons': len(results),
            'matches_found': matches_found,
            'match_rate': (matches_found / len(results) * 100) if results else 0,
            'results': [
                {
                    'database_file': r['database_file'],
                    'result_data': r['result'].to_serializable_dict()
                } for r in results
            ]
        }
        
        with open(result_filepath_json, 'w') as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Case {case_id} processing complete: {len(results)} comparisons, {matches_found} matches")
        flash(f"Analysis complete. Processed {len(results)} comparisons. Found {matches_found} matches.")
        return redirect(url_for('display_results', case_id=case_id))

    except Exception as e:
        logger.error(f"Error in upload_files for case {case_id}: {e}")
        flash(f"An error occurred during analysis: {e}")
        return redirect(url_for('index'))

@app.route('/results/<case_id>')
def display_results(case_id):
    if case_id not in case_results:
        flash("Case results not found.")
        return redirect(url_for('index'))

    data = case_results[case_id]
    
    # FILTER TO SHOW ONLY POSITIVE MATCHES IN THE WEB DISPLAY
    positive_matches = []
    positive_visualizations = []
    
    for i, r in enumerate(data['results']):
        if r['result'].is_match:  # Only include positive matches
            positive_matches.append(r)
            # Get corresponding visualization
            if i < len(data['visualizations']):
                positive_visualizations.append(data['visualizations'][i])
            else:
                positive_visualizations.append(None)
    
    # Create a filtered data structure for display
    display_data = {
        'results': positive_matches,
        'suspect_path': data['suspect_path'],
        'visualizations': positive_visualizations,
        'scan_mode': data['scan_mode'],
        'timestamp': data['timestamp'],
        'total_comparisons': len(data['results']),  # Keep total count for stats
        'total_matches': len(positive_matches)      # Positive matches count
    }
    
    # Debug logging with enhanced consistency check
    matches_count = len(positive_matches)
    logger.info(f"Displaying results for case {case_id}: {len(data['results'])} total, {matches_count} positive matches shown")
    
    # Run consistency check on positive matches only
    if positive_matches:
        debug_result_consistency(f"{case_id}_positive_only", positive_matches)
    
    return render_template('comprehensive_results.html', 
                         case_id=case_id, 
                         data=display_data)

@app.route('/download-json/<filename>')
def serve_result_file(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash("File not found.")
        return redirect(url_for('index'))
    return send_file(filepath, as_attachment=True)

@app.route('/view-visualization/<filename>')
def view_visualization(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash("Visualization not found.")
        return redirect(url_for('index'))
    return send_file(filepath)

@app.route('/clear-case/<case_id>')
def clear_case(case_id):
    """Clear a specific case from memory and optionally delete files."""
    if case_id in case_results:
        del case_results[case_id]
        flash(f"Case {case_id} cleared from memory.")
    else:
        flash("Case not found.")
    return redirect(url_for('index'))

@app.route('/clear-all-cases')
def clear_all_cases():
    """Clear all cases from memory."""
    case_results.clear()
    flash("All cases cleared from memory.")
    return redirect(url_for('index'))

@app.route('/download-comprehensive-pdf/<case_id>')
def download_comprehensive_pdf_report(case_id):
    if case_id not in case_results:
        flash("Case results not found for PDF generation.")
        return redirect(url_for('index'))

    data = case_results[case_id]
    results = data['results']  # PDF INCLUDES ALL RESULTS, NOT JUST POSITIVE MATCHES
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=20, leading=24, alignment=TA_CENTER,
                              spaceAfter=20, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubtitleStyle', fontSize=12, leading=14, alignment=TA_CENTER,
                              spaceAfter=10, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='ReportHeading', fontSize=12, leading=14, fontName='Helvetica-Bold',
                              spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='BodyTextLeft', fontSize=9, leading=11, alignment=TA_LEFT,
                              spaceAfter=4))
    styles.add(ParagraphStyle(name='BodyTextCenter', fontSize=9, leading=11, alignment=TA_CENTER,
                              spaceAfter=4))

    Story = []

    # Header
    Story.append(Paragraph("FORENSIC FINGERPRINT COMPREHENSIVE ANALYSIS REPORT", styles['TitleStyle']))
    Story.append(Spacer(0, 0.2 * inch))

    # Case Information
    Story.append(Paragraph(f"Case ID: {case_id}", styles['ReportHeading']))
    Story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Analysis Mode: {'Database Scan' if data['scan_mode'] else 'Single Comparison'}", styles['BodyTextLeft']))
    Story.append(Spacer(0, 0.15 * inch))

    # Summary Statistics
    total_comparisons = len(results)
    matches_found = sum(1 for r in results if r['result'].is_match)
    match_rate = (matches_found / total_comparisons * 100) if total_comparisons > 0 else 0
    
    Story.append(Paragraph("SUMMARY STATISTICS:", styles['ReportHeading']))
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Comparisons', str(total_comparisons)],
        ['Matches Found', str(matches_found)],
        ['Match Rate', f"{match_rate:.1f}%"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
        ('GRID', (0, 0), (-1, -1), 1, black),
    ]))
    Story.append(summary_table)
    Story.append(Spacer(0, 0.2 * inch))

    # Results table for top matches
    if matches_found > 0:
        Story.append(Paragraph("POSITIVE MATCHES FOUND:", styles['ReportHeading']))
        
        match_data = [['Database File', 'Confidence', 'Match Count', 'Quality Score', 'Geometric Verified']]
        
        # Sort matches by confidence score (highest first)
        sorted_matches = sorted(
            [r for r in results if r['result'].is_match], 
            key=lambda x: x['result'].confidence_score, 
            reverse=True
        )
        
        for r in sorted_matches:
            result = r['result']
            avg_quality = (result.quality_score_img1 + result.quality_score_img2) / 2
            match_data.append([
                r['database_file'],
                f"{result.confidence_score:.3f}",
                str(result.match_count),
                f"{avg_quality:.3f}",
                "YES" if result.geometric_verification else "NO"
            ])
        
        if len(match_data) > 1:
            table = Table(match_data, colWidths=[2.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), green),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), '#F0F0F0'),
                ('GRID', (0, 0), (-1, -1), 1, black),
            ]))
            Story.append(table)
            
            # Add visualizations for positive matches
            Story.append(PageBreak())
            Story.append(Paragraph("MATCH VISUALIZATIONS:", styles['ReportHeading']))
            
            for i, r in enumerate(sorted_matches[:5]):  # Limit to top 5 matches to prevent huge PDFs
                try:
                    # Find corresponding visualization
                    match_index = None
                    for j, result_item in enumerate(results):
                        if (result_item['database_file'] == r['database_file'] and 
                            result_item['result'].confidence_score == r['result'].confidence_score):
                            match_index = j
                            break
                    
                    if match_index is not None and match_index < len(data['visualizations']) and data['visualizations'][match_index]:
                        viz_path = os.path.join(app.config['RESULTS_FOLDER'], data['visualizations'][match_index])
                        if os.path.exists(viz_path):
                            avg_quality = (r['result'].quality_score_img1 + r['result'].quality_score_img2) / 2
                            Story.append(Paragraph(f"Match #{i+1}: {r['database_file']}", styles['ReportHeading']))
                            Story.append(Paragraph(f"Confidence: {r['result'].confidence_score:.3f} | "
                                                 f"Matches: {r['result'].match_count} | "
                                                 f"Quality: {avg_quality:.3f}", styles['BodyTextLeft']))
                            
                            # Add the visualization image
                            img = Image(viz_path, width=6*inch, height=3*inch)
                            Story.append(img)
                            Story.append(Spacer(0, 0.2 * inch))
                            
                            # Add page break between visualizations except for the last one
                            if i < len(sorted_matches[:5]) - 1:
                                Story.append(PageBreak())
                                
                except Exception as img_error:
                    logger.error(f"Error adding visualization to PDF for {r['database_file']}: {img_error}")
                    Story.append(Paragraph(f"Visualization not available for {r['database_file']}", styles['BodyTextLeft']))
                    continue
                    
    else:
        Story.append(Paragraph("NO MATCHES FOUND", styles['ReportHeading']))
        Story.append(Paragraph("The suspect fingerprint did not match any prints in the database.", styles['BodyTextLeft']))
    
    Story.append(PageBreak())

    # Individual result summaries (ALL RESULTS IN PDF - limit to prevent large PDFs)
    Story.append(Paragraph("DETAILED COMPARISON RESULTS:", styles['ReportHeading']))
    Story.append(Paragraph("(All results sorted by confidence)", styles['BodyTextLeft']))
    Story.append(Spacer(0, 0.1 * inch))
    
    # Sort all results by confidence - INCLUDE ALL RESULTS IN PDF
    sorted_results = sorted(results, key=lambda x: x['result'].confidence_score, reverse=True)
    
    # Create detailed results table for ALL results
    detailed_data = [['#', 'Database File', 'Match', 'Confidence', 'Matches', 'Quality', 'Geometric']]
    
    for i, r in enumerate(sorted_results):
        result = r['result']
        avg_quality = (result.quality_score_img1 + result.quality_score_img2) / 2
        
        detailed_data.append([
            str(i+1),
            r['database_file'][:25] + '...' if len(r['database_file']) > 25 else r['database_file'],
            'YES' if result.is_match else 'NO',
            f"{result.confidence_score:.3f}",
            str(result.match_count),
            f"{avg_quality:.3f}",
            'YES' if result.geometric_verification else 'NO'
        ])
    
    detailed_table = Table(detailed_data, colWidths=[0.3*inch, 2.2*inch, 0.5*inch, 0.7*inch, 0.6*inch, 0.6*inch, 0.6*inch])
    detailed_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), '#F8F8F8'),
        ('GRID', (0, 0), (-1, -1), 1, black),
    ]))
    
    # Color code the match column based on YES/NO
    for i in range(1, len(detailed_data)):
        if detailed_data[i][2] == 'YES':
            detailed_table.setStyle(TableStyle([('TEXTCOLOR', (2, i), (2, i), green)]))
        else:
            detailed_table.setStyle(TableStyle([('TEXTCOLOR', (2, i), (2, i), red)]))
    
    Story.append(detailed_table)

    # Add metadata footer
    Story.append(Spacer(0, 0.3 * inch))
    Story.append(Paragraph("TECHNICAL DETAILS:", styles['ReportHeading']))
    Story.append(Paragraph(f"Algorithm Version: {matcher.version}", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Processing Configuration: SIFT Features={forensic_config['sift_features']}, "
                          f"Lowe Ratio={forensic_config['lowe_ratio']}, "
                          f"Min Matches={forensic_config['min_match_count']}", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['BodyTextLeft']))

    try:
        doc.build(Story)
        buffer.seek(0)
        
        # Fix the timestamp variable issue
        timestamp = data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        return send_file(buffer, as_attachment=True, 
                        download_name=f"ForensicReport_{case_id}_{timestamp}.pdf", 
                        mimetype='application/pdf')
    except Exception as pdf_error:
        logger.error(f"PDF generation failed for case {case_id}: {pdf_error}")
        flash(f"Error generating PDF report: {str(pdf_error)}")
        return redirect(url_for('display_results', case_id=case_id))

@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    return {
        'status': 'online',
        'matcher_version': matcher.version,
        'active_cases': len(case_results),
        'timestamp': datetime.now().isoformat()
    }

@app.route('/api/cases')
def api_cases():
    """API endpoint to list all active cases."""
    cases_info = []
    for case_id, data in case_results.items():
        matches_found = sum(1 for r in data['results'] if r['result'].is_match)
        cases_info.append({
            'case_id': case_id,
            'total_comparisons': len(data['results']),
            'matches_found': matches_found,
            'scan_mode': data['scan_mode'],
            'timestamp': data.get('timestamp', 'unknown')
        })
    
    return {'cases': cases_info}

def debug_match_result_class():
    """Debug function to check MatchResult class behavior."""
    logger.info("=== DEBUGGING MatchResult CLASS ===")
    
    try:
        # Create a test result
        test_result = MatchResult(
            is_match=True,
            confidence_score=0.85,
            match_count=25,
            quality_score_img1=0.7,
            quality_score_img2=0.6,
            processing_time=1.2,
            case_id="TEST_001",
            operator_id="DEBUG",
            geometric_verification=True
        )
        
        logger.info(f"Test result is_match: {test_result.is_match} (type: {type(test_result.is_match)})")
        logger.info(f"Boolean check: {test_result.is_match == True}")
        logger.info(f"Boolean conversion: {bool(test_result.is_match)}")
        
        # Test serialization
        try:
            serialized = test_result.to_serializable_dict()
            logger.info(f"Serialized is_match: {serialized.get('is_match')} (type: {type(serialized.get('is_match'))})")
        except Exception as e:
            logger.error(f"Serialization test failed: {e}")
            
    except Exception as e:
        logger.error(f"Debug function failed: {e}")

@app.route('/debug-case/<case_id>')
def debug_case_results(case_id):
    """Debug endpoint to check case result consistency."""
    if case_id not in case_results:
        return {'error': 'Case not found'}, 404
    
    data = case_results[case_id]
    debug_info = []
    
    for i, r in enumerate(data['results']):
        result = r['result']
        debug_info.append({
            'index': i,
            'database_file': r['database_file'],
            'is_match': result.is_match,
            'is_match_type': str(type(result.is_match)),
            'is_match_repr': repr(result.is_match),
            'confidence_score': result.confidence_score,
            'match_count': result.match_count,
            'boolean_tests': {
                'bool_conversion': bool(result.is_match),
                'equality_test': result.is_match == True,
                'identity_test': result.is_match is True
            }
        })
    
    # Run consistency check
    consistent = debug_result_consistency(case_id, data['results'])
    
    return {
        'case_id': case_id,
        'consistent': consistent,
        'total_results': len(data['results']),
        'debug_info': debug_info
    }

if __name__ == '__main__':
    # Run debug check on startup
    debug_match_result_class()
    
    logger.info("Starting Forensic Fingerprint Analysis Web Application")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Database folder: {app.config['DATABASE_FOLDER']}")
    logger.info(f"Results folder: {app.config['RESULTS_FOLDER']}")
    
    # Additional startup logging
    logger.info("=== ENHANCED DEBUGGING ENABLED ===")
    logger.info("Added enhanced validation and consistency checking")
    logger.info("Added debug endpoint: /debug-case/<case_id>")
    logger.info("DISPLAY: Only positive matches shown on web interface")
    logger.info("PDF: All results included in comprehensive report")
    
    app.run(debug=True, host='0.0.0.0', port=8081)