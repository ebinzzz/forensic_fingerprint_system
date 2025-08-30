from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session, jsonify
import os
import json
import secrets
import copy
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from functools import wraps
import hashlib
import uuid

# Database imports (keeping your existing imports)
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
app.config['DATABASE_FOLDER'] = 'database'
app.config['RESULTS_FOLDER'] = 'forensic_results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = secrets.token_hex(16)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

# Database configuration
DATABASE_PATH = 'afis_system.db'

# Ensure all folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize matcher (keeping your existing configuration)
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

# --- Database Setup ---
def update_database_schema():
    """Update the database schema to include new MatchResult fields."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Add missing columns to match_results table
    new_columns = [
        'image1_hash TEXT',
        'image2_hash TEXT', 
        'threshold_used REAL',
        'algorithm_version TEXT',
        'match_timestamp TIMESTAMP'
    ]
    
    for column in new_columns:
        try:
            cursor.execute(f'ALTER TABLE match_results ADD COLUMN {column}')
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                continue  # Column already exists
            else:
                logger.error(f"Error adding column {column}: {e}")
    
    conn.commit()
    conn.close()
    logger.info("Database schema updated successfully")

def init_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            badge_id TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            department TEXT,
            role TEXT DEFAULT 'examiner',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Cases table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT UNIQUE NOT NULL,
            operator_id TEXT NOT NULL,
            operator_name TEXT,
            analysis_mode TEXT NOT NULL,
            total_comparisons INTEGER DEFAULT 0,
            matches_found INTEGER DEFAULT 0,
            match_rate REAL DEFAULT 0.0,
            status TEXT DEFAULT 'completed',
            suspect_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_time REAL,
            notes TEXT
        )
    ''')
    
    # Match results table WITH visualization_path
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            database_file TEXT NOT NULL,
            is_match BOOLEAN NOT NULL,
            confidence_score REAL NOT NULL,
            match_count INTEGER NOT NULL,
            quality_score_img1 REAL,
            quality_score_img2 REAL,
            processing_time REAL,
            geometric_verification BOOLEAN,
            visualization_path TEXT,
            image1_hash TEXT,
            image2_hash TEXT,
            threshold_used REAL,
            algorithm_version TEXT,
            match_timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases (case_id)
        )
    ''')
    
    # System logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create default admin user if no users exist
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        admin_hash = generate_password_hash('admin123')
        cursor.execute('''
            INSERT INTO users (username, badge_id, password_hash, full_name, department, role)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('admin', 'ADMIN001', admin_hash, 'System Administrator', 'IT', 'admin'))
        
        # Create sample examiner
        examiner_hash = generate_password_hash('examiner123')
        cursor.execute('''
            INSERT INTO users (username, badge_id, password_hash, full_name, department, role)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('examiner1', 'EX001', examiner_hash, 'John Smith', 'Forensics', 'examiner'))
    
    conn.commit()
    conn.close()
    
    # Update schema for new MatchResult requirements and visualization paths
    update_database_schema()
    update_database_schema_for_visualizations()
    
    logger.info("Database initialized successfully")

def update_database_schema_for_visualizations():
    """Update the database schema to include visualization_path column."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check if visualization_path column exists
    cursor.execute("PRAGMA table_info(match_results)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'visualization_path' not in columns:
        try:
            cursor.execute('ALTER TABLE match_results ADD COLUMN visualization_path TEXT')
            conn.commit()
            logger.info("Added visualization_path column to match_results table")
        except sqlite3.OperationalError as e:
            logger.error(f"Error adding visualization_path column: {e}")
    else:
        logger.info("visualization_path column already exists")
    
def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def log_user_action(user_id, action, details=None, ip_address=None):
    """Log user actions to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO system_logs (user_id, action, details, ip_address)
        VALUES (?, ?, ?, ?)
    ''', (user_id, action, details, ip_address))
    conn.commit()
    conn.close()

# --- Authentication Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        user = conn.execute('SELECT role FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        
        if not user or user['role'] != 'admin':
            flash('Administrative privileges required.')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember_me = request.form.get('remember_me')
        
        # Add debug logging
        print(f"DEBUG: Login attempt - Username: {username}, Password length: {len(password) if password else 0}")
        
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
        
        conn = get_db_connection()
        try:
            user = conn.execute('''
                SELECT id, username, badge_id, password_hash, full_name, role, is_active 
                FROM users WHERE (username = ? OR badge_id = ?) AND is_active = 1
            ''', (username, username)).fetchone()
            
            print(f"DEBUG: User found: {user is not None}")
            if user:
                print(f"DEBUG: User details - ID: {user['id']}, Username: {user['username']}, Role: {user['role']}")
            
            if user and check_password_hash(user['password_hash'], password):
                print(f"DEBUG: Password check passed for user: {user['username']}")
                
                # Set session permanent before setting session data
                if remember_me:
                    session.permanent = True
                    app.permanent_session_lifetime = timedelta(hours=8)
                
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['badge_id'] = user['badge_id']
                session['full_name'] = user['full_name']
                session['role'] = user['role']
                
                print(f"DEBUG: Session data set - User ID: {session.get('user_id')}, Role: {session.get('role')}")
                
                # Update last login
                cursor = conn.cursor()
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
                conn.commit()
                print(f"DEBUG: Last login updated for user: {user['username']}")
                
                # Log successful login
                try:
                    log_user_action(user['username'], 'login_success', 
                                  f"Role: {user['role']}", request.remote_addr)
                    print(f"DEBUG: User action logged successfully")
                except Exception as log_error:
                    print(f"DEBUG: Error logging user action: {log_error}")
                    # Don't fail login due to logging error
                
                flash(f'Welcome back, {user["full_name"]}!', 'success')
                print(f"DEBUG: About to redirect to dashboard")
                
                # Check if dashboard route exists and is accessible
                try:
                    dashboard_url = url_for('dashboard')
                    print(f"DEBUG: Dashboard URL generated: {dashboard_url}")
                    return redirect(dashboard_url)
                except Exception as redirect_error:
                    print(f"DEBUG: Error generating dashboard URL: {redirect_error}")
                    flash('Login successful, but dashboard unavailable. Please contact IT.', 'error')
                    return render_template('login.html')
                    
            else:
                print(f"DEBUG: Authentication failed - User found: {user is not None}, Password check: {'N/A' if not user else 'Failed'}")
                # Log failed login attempt
                try:
                    log_user_action(username, 'login_failed', 
                                  f"Failed login attempt", request.remote_addr)
                except Exception as log_error:
                    print(f"DEBUG: Error logging failed attempt: {log_error}")
                
                flash('Invalid credentials. Please try again.', 'error')
        
        except Exception as db_error:
            print(f"DEBUG: Database error during login: {db_error}")
            flash('System error during login. Please try again.', 'error')
        
        finally:
            conn.close()
    
    return render_template('login.html')

# Additional debugging route to check session
@app.route('/debug-session')
def debug_session():
    """Debugging route to check session data (remove in production)"""
    if app.debug:  # Only available in debug mode
        return {
            'session_data': dict(session),
            'user_logged_in': 'user_id' in session,
            'permanent_session': session.permanent
        }
    else:
        return "Debug mode only", 404

@app.route('/logout')
def logout():
    if 'username' in session:
        log_user_action(session['username'], 'logout', None, request.remote_addr)
    
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')

# --- Dashboard and Main Routes ---
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    
    # Get user's recent cases
    user_cases = conn.execute('''
        SELECT case_id, analysis_mode, total_comparisons, matches_found, 
               match_rate, status, created_at
        FROM cases WHERE operator_id = ?
        ORDER BY created_at DESC LIMIT 10
    ''', (session['badge_id'],)).fetchall()
    
    # Get system statistics
    total_cases = conn.execute('SELECT COUNT(*) as count FROM cases').fetchone()['count']
    total_matches = conn.execute('SELECT SUM(matches_found) as total FROM cases').fetchone()['total'] or 0
    
    # Get recent system activity
    recent_activity = conn.execute('''
        SELECT action, details, timestamp FROM system_logs 
        ORDER BY timestamp DESC LIMIT 5
    ''').fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                         user_cases=user_cases,
                         total_cases=total_cases,
                         total_matches=total_matches,
                         recent_activity=recent_activity)

@app.route('/new_case')
@login_required
def new_case():
    return render_template('index.html')

@app.route('/')
def index1():
    return render_template('index1.html')



# --- Case Management ---
def save_case_to_database_with_viz(case_id, results, operator_id, analysis_mode, processing_start_time):
    """Save case and match results to database with visualization paths."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get operator name
        operator = conn.execute('SELECT full_name FROM users WHERE badge_id = ?', (operator_id,)).fetchone()
        operator_name = operator['full_name'] if operator else operator_id
        
        # Calculate statistics
        total_comparisons = len(results)
        matches_found = sum(1 for r in results if r['result'].is_match)
        match_rate = (matches_found / total_comparisons * 100) if total_comparisons > 0 else 0
        processing_time = (datetime.now() - processing_start_time).total_seconds()
        
        # Insert case record
        cursor.execute('''
            INSERT INTO cases (case_id, operator_id, operator_name, analysis_mode, 
                             total_comparisons, matches_found, match_rate, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (case_id, operator_id, operator_name, analysis_mode, 
              total_comparisons, matches_found, match_rate, processing_time))
        
        # Insert individual match results with visualization paths
        for result_data in results:
            result = result_data['result']
            viz_path = result_data.get('visualization', '')  # Get visualization filename
            
            cursor.execute('''
                INSERT INTO match_results (case_id, database_file, is_match, confidence_score,
                                         match_count, quality_score_img1, quality_score_img2,
                                         processing_time, geometric_verification, image1_hash,
                                         image2_hash, threshold_used, algorithm_version, 
                                         match_timestamp, visualization_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (case_id, result_data['database_file'], result.is_match, 
                  result.confidence_score, result.match_count, result.quality_score_img1,
                  result.quality_score_img2, result.processing_time, result.geometric_verification,
                  getattr(result, 'image1_hash', ''), getattr(result, 'image2_hash', ''),
                  getattr(result, 'threshold_used', 0.0), getattr(result, 'algorithm_version', 'v1.0'),
                  getattr(result, 'timestamp', datetime.now().isoformat()), viz_path))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Case {case_id} with visualizations saved to database successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error saving case {case_id} to database: {e}")
        return False

def load_case_from_database(case_id):
    """Load case results from database with visualization paths."""
    try:
        conn = get_db_connection()
        
        # Get case info
        case = conn.execute('SELECT * FROM cases WHERE case_id = ?', (case_id,)).fetchone()
        if not case:
            conn.close()
            return None
        
        # Get match results with visualization paths
        matches = conn.execute('''
            SELECT * FROM match_results WHERE case_id = ? ORDER BY confidence_score DESC
        ''', (case_id,)).fetchall()
        
        conn.close()
        
        # Reconstruct results format
        results = []
        visualizations = []
        
        for match in matches:
            match_dict = dict(match)
            
            # Try to create MatchResult with all parameters
            try:
                result = MatchResult(
                    is_match=bool(match_dict['is_match']),
                    confidence_score=match_dict['confidence_score'],
                    match_count=match_dict['match_count'],
                    quality_score_img1=match_dict['quality_score_img1'],
                    quality_score_img2=match_dict['quality_score_img2'],
                    processing_time=match_dict['processing_time'],
                    case_id=case_id,
                    operator_id=case['operator_id'],
                    geometric_verification=bool(match_dict['geometric_verification']),
                    image1_hash=match_dict.get('image1_hash', ''),
                    image2_hash=match_dict.get('image2_hash', ''),
                    threshold_used=match_dict.get('threshold_used', 0.0),
                    algorithm_version=match_dict.get('algorithm_version', 'v1.0'),
                    timestamp=match_dict.get('match_timestamp', datetime.now().isoformat())
                )
            except TypeError as e:
                logger.warning(f"Failed to create MatchResult with new constructor: {e}")
                # Create a minimal compatible object for old records
                class CompatibleMatchResult:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                
                result = CompatibleMatchResult(
                    is_match=bool(match_dict['is_match']),
                    confidence_score=match_dict['confidence_score'],
                    match_count=match_dict['match_count'],
                    quality_score_img1=match_dict['quality_score_img1'],
                    quality_score_img2=match_dict['quality_score_img2'],
                    processing_time=match_dict['processing_time'],
                    geometric_verification=bool(match_dict['geometric_verification'])
                )
            
            results.append({
                'result': result,
                'database_file': match_dict['database_file'],
                'database_path': os.path.join(app.config['DATABASE_FOLDER'], match_dict['database_file'])
            })
            
            # Add visualization filename to list
            viz_file = match_dict.get('visualization_path', '')
            visualizations.append(viz_file if viz_file else None)
        
        return {
            'results': results,
            'scan_mode': case['analysis_mode'] == 'database',
            'timestamp': case['created_at'],
            'case_info': dict(case),
            'visualizations': visualizations  # Now contains actual filenames
        }
        
    except Exception as e:
        logger.error(f"Error loading case {case_id} from database: {e}")
        return None

# --- Your existing utility functions (keeping them) ---
def get_database_images():
    """Get all image files from the database folder."""
    supported_extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.pgm', '.ppm']
    database_images = []
    
    for ext in supported_extensions:
        pattern = os.path.join(app.config['DATABASE_FOLDER'], f"*{ext}")
        database_images.extend(glob.glob(pattern))
        pattern = os.path.join(app.config['DATABASE_FOLDER'], f"*{ext.upper()}")
        database_images.extend(glob.glob(pattern))
    
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
    
    logger.info(f"\n=== CONSISTENCY CHECK COMPLETE ===")
    logger.info(f"Total inconsistencies found: {inconsistencies_found}")
    return inconsistencies_found == 0

def validate_match_result_enhanced(result, database_file):
    """Enhanced validation with detailed type and value checking."""
    try:
        logger.info(f"=== ENHANCED VALIDATION: {database_file} ===")
        
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
        
        logger.info(f"VALIDATION PASSED for {database_file}")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced validation failed for {database_file}: {e}")
        return False

def create_comparison_visualization(suspect_path, database_path, result):
    """Create a visualization showing the comparison between two fingerprints."""
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
            combined = np.hstack([cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), 
                                cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)])
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
        
        is_match = bool(result.is_match)
        
        # Enhanced validation logging
        logger.info(f"VISUALIZATION - File: {os.path.basename(database_path)}")
        logger.info(f"VISUALIZATION - result.is_match: {result.is_match} (type: {type(result.is_match)})")
        logger.info(f"VISUALIZATION - converted is_match: {is_match}")
        
        # Draw matches
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
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        if is_match:
            status_text = "MATCH: YES"
            status_color = (0, 255, 0)
        else:
            status_text = "MATCH: NO" 
            status_color = (0, 0, 255)
            
        cv2.putText(img_matches, status_text, (10, 30), font, font_scale, status_color, thickness)
        cv2.putText(img_matches, f"Confidence: {result.confidence_score:.3f}", (10, 60), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img_matches, f"Matches: {result.match_count}", (10, 90), font, font_scale, (255, 255, 255), thickness)
        
        avg_quality = (result.quality_score_img1 + result.quality_score_img2) / 2
        cv2.putText(img_matches, f"Quality: {avg_quality:.3f}", (10, 120), font, font_scale, (255, 255, 255), thickness)
        
        return img_matches
        
    except Exception as e:
        logger.error(f"Error creating visualization for {database_path}: {e}")
        return None

# --- Modified Upload Route ---
@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    if 'file1' not in request.files:
        flash('Please select a suspect fingerprint image.')
        return redirect(url_for('new_case'))

    file1 = request.files['file1']
    file2 = request.files.get('file2')
    case_id = request.form.get('case_id')
    analysis_mode = request.form.get('analysis_mode', 'database')

    if file1.filename == '' or not case_id:
        flash('Case ID and suspect fingerprint are required.')
        return redirect(url_for('new_case'))

    # Check if case already exists in database
    conn = get_db_connection()
    existing_case = conn.execute('SELECT case_id FROM cases WHERE case_id = ?', (case_id,)).fetchone()
    if existing_case:
        conn.close()
        flash(f'Case {case_id} already exists. Please use a different case ID.')
        return redirect(url_for('new_case'))
    conn.close()

    processing_start_time = datetime.now()

    try:
        # Save suspect fingerprint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename1 = f"{case_id}_{timestamp}_suspect_{file1.filename}"
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file1.save(filepath1)

        results = []
        operator_id = session['badge_id']

        if file2 and file2.filename != '':
            # Single comparison mode
            filename2 = f"{case_id}_{timestamp}_database_{file2.filename}"
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file2.save(filepath2)

            logger.info(f"Starting single comparison for case {case_id}")
            result = matcher.match_fingerprints(filepath1, filepath2, case_id, operator_id)
            
            if not validate_match_result_enhanced(result, filename2):
                logger.error(f"Validation failed for {filename2}")
                flash("Validation error occurred during analysis.")
                return redirect(url_for('new_case'))
            
            result_copy = copy.deepcopy(result)
            
            # Create visualization BEFORE appending to results
            visualization = create_comparison_visualization(filepath1, filepath2, result_copy)
            viz_filename = None
            if visualization is not None:
                viz_filename = f"{case_id}_single_comparison.png"
                viz_path = os.path.join(app.config['RESULTS_FOLDER'], viz_filename)
                cv2.imwrite(viz_path, visualization)
                logger.info(f"Visualization saved: {viz_path}")
            
            results.append({
                'result': result_copy,
                'database_file': filename2,
                'database_path': filepath2,
                'visualization': viz_filename  # Add visualization filename to result
            })

        else:
            # Database scanning mode
            database_images = get_database_images()
            
            if not database_images:
                flash('No images found in the database folder.')
                return redirect(url_for('new_case'))

            logger.info(f'Starting database scan for case {case_id} against {len(database_images)} images')
            
            for i, db_image_path in enumerate(database_images):
                try:
                    db_filename = os.path.basename(db_image_path)
                    sub_case_id = f"{case_id}_{i+1:03d}"
                    
                    logger.info(f"Processing {i+1}/{len(database_images)}: {db_filename}")
                    result = matcher.match_fingerprints(filepath1, db_image_path, sub_case_id, operator_id)
                    
                    if not validate_match_result_enhanced(result, db_filename):
                        logger.error(f"Validation failed for {db_filename} - skipping")
                        continue
                    
                    result_copy = copy.deepcopy(result)
                    
                    # Create visualization
                    visualization = create_comparison_visualization(filepath1, db_image_path, result_copy)
                    viz_filename = None
                    if visualization is not None:
                        viz_filename = f"{sub_case_id}_comparison.png"
                        viz_path = os.path.join(app.config['RESULTS_FOLDER'], viz_filename)
                        cv2.imwrite(viz_path, visualization)
                        logger.info(f"Visualization saved: {viz_path}")

                    results.append({
                        'result': result_copy,
                        'database_file': db_filename,
                        'database_path': db_image_path,
                        'visualization': viz_filename  # Add visualization filename to result
                    })

                except Exception as e:
                    logger.error(f"Error processing database image {db_image_path}: {e}")
                    continue

        # Save to database with visualization paths
        if save_case_to_database_with_viz(case_id, results, operator_id, analysis_mode, processing_start_time):
            # Log the action
            matches_found = sum(1 for r in results if r['result'].is_match)
            log_user_action(session['username'], 'case_processed', 
                          f"Case: {case_id}, Comparisons: {len(results)}, Matches: {matches_found}")
            
            flash(f"Analysis complete. Case saved to database. Found {matches_found} matches.")
        else:
            flash("Warning: Case processing completed but database save failed.")

        return redirect(url_for('display_results', case_id=case_id))

    except Exception as e:
        logger.error(f"Error in upload_files for case {case_id}: {e}")
        flash(f"An error occurred during analysis: {e}")
        return redirect(url_for('new_case'))


# --- Results Display Routes ---
@app.route('/results/<case_id>')
@login_required
def display_results(case_id):
    # First try to load from database
    data = load_case_from_database(case_id)
    
    if not data:
        flash("Case results not found.")
        return redirect(url_for('dashboard'))

    # Check if user has access to this case
    if session['role'] != 'admin' and data['case_info']['operator_id'] != session['badge_id']:
        flash("Access denied. You can only view your own cases.")
        return redirect(url_for('dashboard'))
  
    # Log the view action
    log_user_action(session['username'], 'case_viewed', f"Case: {case_id}")
    
    return render_template('comprehensive_results.html', 
                         case_id=case_id, 
                         data=data)

# --- Case Management Routes ---
@app.route('/cases')
@login_required
def list_cases():
    conn = get_db_connection()
    
    if session['role'] == 'admin':
        # Admin can see all cases
        cases = conn.execute('''
            SELECT c.*, COUNT(mr.id) as total_results
            FROM cases c
            LEFT JOIN match_results mr ON c.case_id = mr.case_id
            GROUP BY c.case_id
            ORDER BY c.created_at DESC
        ''').fetchall()
    else:
        # Regular users see only their cases
        cases = conn.execute('''
            SELECT c.*, COUNT(mr.id) as total_results
            FROM cases c
            LEFT JOIN match_results mr ON c.case_id = mr.case_id
            WHERE c.operator_id = ?
            GROUP BY c.case_id
            ORDER BY c.created_at DESC
        ''', (session['badge_id'],)).fetchall()
    
    conn.close()
    return render_template('cases_list.html', cases=cases)

@app.route('/case/<case_id>/delete', methods=['POST'])
@login_required
def delete_case(case_id):
    conn = get_db_connection()
    
    # Check if case exists and user has permission
    case = conn.execute('SELECT operator_id FROM cases WHERE case_id = ?', (case_id,)).fetchone()
    
    if not case:
        flash("Case not found.")
        conn.close()
        return redirect(url_for('list_cases'))
    
    if session['role'] != 'admin' and case['operator_id'] != session['badge_id']:
        flash("Access denied. You can only delete your own cases.")
        conn.close()
        return redirect(url_for('list_cases'))
    
    try:
        # Delete match results first (foreign key constraint)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM match_results WHERE case_id = ?', (case_id,))
        cursor.execute('DELETE FROM cases WHERE case_id = ?', (case_id,))
        conn.commit()
        
        # Log the deletion
        log_user_action(session['username'], 'case_deleted', f"Case: {case_id}")
        
        flash(f"Case {case_id} deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting case {case_id}: {e}")
        flash("Error deleting case.")
    
    conn.close()
    return redirect(url_for('list_cases'))

# --- User Management (Admin only) ---
@app.route('/users')
@admin_required
def list_users():
    conn = get_db_connection()
    users = conn.execute('''
        SELECT u.*, COUNT(c.case_id) as total_cases
        FROM users u
        LEFT JOIN cases c ON u.badge_id = c.operator_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    ''').fetchall()
    conn.close()
    return render_template('users_list.html', users=users)

@app.route('/users/create', methods=['GET', 'POST'])
@admin_required
def create_user():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        badge_id = request.form.get('badge_id', '').strip()
        password = request.form.get('password', '')
        full_name = request.form.get('full_name', '').strip()
        department = request.form.get('department', '').strip()
        role = request.form.get('role', 'examiner')
        
        if not all([username, badge_id, password, full_name]):
            flash('All required fields must be filled.')
            return render_template('create_user.html')
        
        conn = get_db_connection()
        
        # Check for existing username or badge_id
        existing = conn.execute('''
            SELECT COUNT(*) as count FROM users 
            WHERE username = ? OR badge_id = ?
        ''', (username, badge_id)).fetchone()
        
        if existing['count'] > 0:
            flash('Username or Badge ID already exists.')
            conn.close()
            return render_template('create_user.html')
        
        try:
            password_hash = generate_password_hash(password)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, badge_id, password_hash, full_name, department, role)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, badge_id, password_hash, full_name, department, role))
            conn.commit()
            
            log_user_action(session['username'], 'user_created', 
                          f"Created user: {username} ({badge_id})")
            
            flash(f"User {username} created successfully.")
            return redirect(url_for('list_users'))
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            flash("Error creating user.")
        finally:
            conn.close()
    
    return render_template('create_user.html')

# --- Report Generation Routes ---
@app.route('/download-json/<filename>')
@login_required
def serve_result_file(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash("File not found.")
        return redirect(url_for('dashboard'))
    return send_file(filepath, as_attachment=True)

@app.route('/view-visualization/<filename>')
@login_required
def view_visualization(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash("Visualization not found.")
        return redirect(url_for('dashboard'))
    return send_file(filepath)

@app.route('/download-comprehensive-pdf/<case_id>')
@login_required
def download_comprehensive_pdf_report(case_id):
    # Load case from database
    data = load_case_from_database(case_id)
    
    if not data:
        flash("Case results not found for PDF generation.")
        return redirect(url_for('dashboard'))
    
    # Check access permissions
    if session['role'] != 'admin' and data['case_info']['operator_id'] != session['badge_id']:
        flash("Access denied. You can only download reports for your own cases.")
        return redirect(url_for('dashboard'))
    
    results = data['results']
    visualizations = data.get('visualizations', [])
    
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
    styles.add(ParagraphStyle(name='CenteredText', fontSize=10, leading=12, alignment=TA_CENTER,
                              spaceAfter=6))

    Story = []

    # Header
    Story.append(Paragraph("FORENSIC FINGERPRINT COMPREHENSIVE ANALYSIS REPORT", styles['TitleStyle']))
    Story.append(Spacer(0, 0.2 * inch))

    # Case Information
    case_info = data['case_info']
    Story.append(Paragraph(f"Case ID: {case_id}", styles['ReportHeading']))
    Story.append(Paragraph(f"Examiner: {case_info['operator_name']} ({case_info['operator_id']})", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Analysis Mode: {case_info['analysis_mode'].title()}", styles['BodyTextLeft']))
    Story.append(Spacer(0, 0.15 * inch))

    # Summary Statistics
    total_comparisons = len(results)
    matches_found = sum(1 for r in results if r['result'].is_match)
    match_rate = (matches_found / total_comparisons * 100) if total_comparisons > 0 else 0
    
    Story.append(Paragraph("SUMMARY STATISTICS:", styles['ReportHeading']))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Comparisons', str(total_comparisons)],
        ['Matches Found', str(matches_found)],
        ['Match Rate', f"{match_rate:.1f}%"],
        ['Processing Time', f"{case_info['processing_time']:.2f}s"]
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

    # Results table for positive matches with visualizations
    if matches_found > 0:
        Story.append(Paragraph("POSITIVE MATCHES FOUND:", styles['ReportHeading']))
        
        # Get sorted matches
        sorted_matches = sorted(
            [r for r in results if r['result'].is_match], 
            key=lambda x: x['result'].confidence_score, 
            reverse=True
        )
        
        for match_idx, match_result in enumerate(sorted_matches):
            result = match_result['result']
            database_file = match_result['database_file']
            
            # Match details table
            avg_quality = (result.quality_score_img1 + result.quality_score_img2) / 2
            
            match_data = [
                ['Database File', database_file],
                ['Confidence Score', f"{result.confidence_score:.3f}"],
                ['Match Count', str(result.match_count)],
                ['Average Quality Score', f"{avg_quality:.3f}"],
                ['Geometric Verification', "YES" if result.geometric_verification else "NO"],
                ['Processing Time', f"{result.processing_time:.2f}s"]
            ]
            
            match_table = Table(match_data, colWidths=[2*inch, 3*inch])
            match_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), green),
                ('TEXTCOLOR', (0, 0), (0, -1), black),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (1, 0), (1, -1), '#F0F0F0'),
                ('GRID', (0, 0), (-1, -1), 1, black),
            ]))
            Story.append(match_table)
            Story.append(Spacer(0, 0.1 * inch))
            
            # Add visualization if available
            # Find the original index of this match in the results list
            original_index = None
            for idx, original_result in enumerate(results):
                if (original_result['database_file'] == database_file and 
                    original_result['result'].confidence_score == result.confidence_score):
                    original_index = idx
                    break
            
            if (original_index is not None and 
                original_index < len(visualizations) and 
                visualizations[original_index]):
                
                viz_filename = visualizations[original_index]
                viz_path = os.path.join(app.config['RESULTS_FOLDER'], viz_filename)
                
                if os.path.exists(viz_path):
                    try:
                        # Create visualization paragraph
                        Story.append(Paragraph(f"Visual Analysis for {database_file}:", styles['CenteredText']))
                        
                        # Add the image with appropriate sizing
                        img = Image(viz_path)
                        
                        # Scale image to fit page width while maintaining aspect ratio
                        img_width, img_height = img.imageWidth, img.imageHeight
                        max_width = 6.5 * inch  # Available width on page
                        max_height = 4 * inch   # Maximum height for visualization
                        
                        # Calculate scaling factor
                        width_ratio = max_width / img_width
                        height_ratio = max_height / img_height
                        scale_factor = min(width_ratio, height_ratio)
                        
                        img.drawWidth = img_width * scale_factor
                        img.drawHeight = img_height * scale_factor
                        
                        Story.append(img)
                        Story.append(Spacer(0, 0.2 * inch))
                        
                        logger.info(f"Added visualization {viz_filename} to PDF report")
                        
                    except Exception as img_error:
                        logger.error(f"Error adding visualization {viz_filename} to PDF: {img_error}")
                        Story.append(Paragraph(f"[Visualization for {database_file} could not be loaded]", 
                                             styles['CenteredText']))
                        Story.append(Spacer(0, 0.1 * inch))
                else:
                    logger.warning(f"Visualization file not found: {viz_path}")
                    Story.append(Paragraph(f"[Visualization file not found for {database_file}]", 
                                         styles['CenteredText']))
                    Story.append(Spacer(0, 0.1 * inch))
            
            # Add page break between matches (except for the last one)
            if match_idx < len(sorted_matches) - 1:
                Story.append(PageBreak())
    
    else:
        Story.append(Paragraph("NO MATCHES FOUND", styles['ReportHeading']))
        Story.append(Paragraph("The suspect fingerprint did not match any prints in the database.", styles['BodyTextLeft']))

    # Add comprehensive results section for database scan mode
    if data.get('scan_mode', False) and len(results) > matches_found:
        Story.append(PageBreak())
        Story.append(Paragraph("COMPLETE ANALYSIS RESULTS:", styles['ReportHeading']))
        Story.append(Paragraph(f"All {total_comparisons} comparisons performed:", styles['BodyTextLeft']))
        Story.append(Spacer(0, 0.1 * inch))
        
        # Create summary table of all results
        all_results_data = [['Database File', 'Match', 'Confidence', 'Match Count', 'Quality']]
        
        for result_item in results:
            result = result_item['result']
            avg_quality = (result.quality_score_img1 + result.quality_score_img2) / 2
            all_results_data.append([
                result_item['database_file'][:30] + ('...' if len(result_item['database_file']) > 30 else ''),
                "YES" if result.is_match else "NO",
                f"{result.confidence_score:.3f}",
                str(result.match_count),
                f"{avg_quality:.2f}"
            ])
        
        # Split into chunks if too many results
        chunk_size = 25
        for i in range(0, len(all_results_data), chunk_size):
            chunk = all_results_data[i:min(i+chunk_size, len(all_results_data))]
            if i > 0:
                # Add header row for continuation
                chunk.insert(0, all_results_data[0])
            
            results_table = Table(chunk, colWidths=[2.2*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.6*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#457b9d'),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), '#F8F9FA'),
                ('GRID', (0, 0), (-1, -1), 1, black),
                # Highlight matches in green
                ('BACKGROUND', (0, 1), (-1, -1), '#d4edda'),  # Light green for matches
            ]))
            
            # Color code matches vs non-matches
            for row_idx, row_data in enumerate(chunk[1:], 1):  # Skip header
                if row_data[1] == "NO":  # No match
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, row_idx), (-1, row_idx), '#f8d7da')  # Light red
                    ]))
            
            Story.append(results_table)
            
            if i + chunk_size < len(all_results_data):
                Story.append(PageBreak())
            else:
                Story.append(Spacer(0, 0.2 * inch))

    # Add metadata footer
    Story.append(Spacer(0, 0.3 * inch))
    Story.append(Paragraph("TECHNICAL DETAILS:", styles['ReportHeading']))
    Story.append(Paragraph(f"Algorithm Version: {matcher.version}", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Generated by: {session['full_name']} ({session['badge_id']})", styles['BodyTextLeft']))
    Story.append(Paragraph(f"Report Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyTextLeft']))
    
    if visualizations:
        viz_count = sum(1 for v in visualizations if v is not None)
        Story.append(Paragraph(f"Visualizations Included: {viz_count} of {len(visualizations)}", styles['BodyTextLeft']))

    try:
        doc.build(Story)
        buffer.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(buffer, as_attachment=True, 
                        download_name=f"ForensicReport_{case_id}_{timestamp}.pdf", 
                        mimetype='application/pdf')
    except Exception as pdf_error:
        logger.error(f"PDF generation failed for case {case_id}: {pdf_error}")
        flash(f"Error generating PDF report: {str(pdf_error)}")
        return redirect(url_for('display_results', case_id=case_id))


# Helper function to optimize visualization images for PDF
def optimize_image_for_pdf(image_path, max_size_kb=500):
    """
    Optimize visualization image for PDF inclusion to reduce file size.
    Returns path to optimized image or original if optimization fails.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # Get original file size
        original_size = os.path.getsize(image_path)
        if original_size <= max_size_kb * 1024:
            return image_path  # Already small enough
        
        # Create optimized version
        optimized_path = image_path.replace('.png', '_optimized.jpg')
        
        # Resize if too large
        height, width = img.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Save as JPEG with compression
        cv2.imwrite(optimized_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Check if optimization was successful
        if os.path.exists(optimized_path) and os.path.getsize(optimized_path) < original_size:
            logger.info(f"Optimized {image_path} from {original_size} to {os.path.getsize(optimized_path)} bytes")
            return optimized_path
        else:
            # Cleanup failed optimization
            if os.path.exists(optimized_path):
                os.remove(optimized_path)
            return image_path
            
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {e}")
        return image_path
# --- API Routes ---
@app.route('/api/status')
@login_required
def api_status():
    conn = get_db_connection()
    total_cases = conn.execute('SELECT COUNT(*) as count FROM cases').fetchone()['count']
    total_users = conn.execute('SELECT COUNT(*) as count FROM users WHERE is_active = 1').fetchone()['count']
    conn.close()
    
    return {
        'status': 'online',
        'matcher_version': matcher.version,
        'total_cases': total_cases,
        'total_users': total_users,
        'current_user': session['username'],
        'timestamp': datetime.now().isoformat()
    }

@app.route('/api/cases')
@login_required
def api_cases():
    conn = get_db_connection()
    
    if session['role'] == 'admin':
        cases = conn.execute('''
            SELECT case_id, operator_name, analysis_mode, total_comparisons, 
                   matches_found, match_rate, created_at
            FROM cases ORDER BY created_at DESC
        ''').fetchall()
    else:
        cases = conn.execute('''
            SELECT case_id, operator_name, analysis_mode, total_comparisons, 
                   matches_found, match_rate, created_at
            FROM cases WHERE operator_id = ? ORDER BY created_at DESC
        ''', (session['badge_id'],)).fetchall()
    
    conn.close()
    
    return {'cases': [dict(case) for case in cases]}

# --- Settings and Profile ---
@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    # Get user statistics
    user_stats = conn.execute('''
        SELECT COUNT(*) as total_cases,
               SUM(matches_found) as total_matches,
               AVG(processing_time) as avg_processing_time
        FROM cases WHERE operator_id = ?
    ''', (session['badge_id'],)).fetchone()
    
    conn.close()
    return render_template('profile.html', user=user, stats=user_stats)

@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    full_name = request.form.get('full_name', '').strip()
    department = request.form.get('department', '').strip()
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    
    if not full_name:
        flash('Full name is required.')
        return redirect(url_for('profile'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if new_password:
            # Verify current password
            user = conn.execute('SELECT password_hash FROM users WHERE id = ?', 
                              (session['user_id'],)).fetchone()
            
            if not check_password_hash(user['password_hash'], current_password):
                flash('Current password is incorrect.')
                conn.close()
                return redirect(url_for('profile'))
            
            # Update with new password
            new_hash = generate_password_hash(new_password)
            cursor.execute('''
                UPDATE users SET full_name = ?, department = ?, password_hash = ?
                WHERE id = ?
            ''', (full_name, department, new_hash, session['user_id']))
            
            log_user_action(session['username'], 'password_changed')
        else:
            # Update without password change
            cursor.execute('''
                UPDATE users SET full_name = ?, department = ?
                WHERE id = ?
            ''', (full_name, department, session['user_id']))
        
        conn.commit()
        session['full_name'] = full_name
        
        log_user_action(session['username'], 'profile_updated')
        flash('Profile updated successfully.')
        
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        flash('Error updating profile.')
    
    conn.close()
    return redirect(url_for('profile'))

# --- System Administration ---
@app.route('/admin')
@admin_required
def admin_panel():
    conn = get_db_connection()
    
    # System statistics
    stats = {
        'total_users': conn.execute('SELECT COUNT(*) as count FROM users WHERE is_active = 1').fetchone()['count'],
        'total_cases': conn.execute('SELECT COUNT(*) as count FROM cases').fetchone()['count'],
        'total_matches': conn.execute('SELECT SUM(matches_found) as total FROM cases').fetchone()['total'] or 0,
        'recent_cases': conn.execute('SELECT COUNT(*) as count FROM cases WHERE created_at >= date("now", "-7 days")').fetchone()['count'],
        'database_images': len(get_database_images())
    }
    
    # Recent activity
    recent_logs = conn.execute('''
        SELECT user_id, action, details, timestamp 
        FROM system_logs 
        ORDER BY timestamp DESC LIMIT 20
    ''').fetchall()
    
    # Top users by case count
    top_users = conn.execute('''
        SELECT u.full_name, u.badge_id, COUNT(c.case_id) as case_count
        FROM users u
        LEFT JOIN cases c ON u.badge_id = c.operator_id
        WHERE u.is_active = 1
        GROUP BY u.id
        ORDER BY case_count DESC LIMIT 5
    ''').fetchall()
    
    conn.close()
    
    return render_template('admin_panel.html', 
                         stats=stats, 
                         recent_logs=recent_logs,
                         top_users=top_users)

@app.route('/users/<int:user_id>/activate', methods=['POST'])
@admin_required
def activate_user(user_id):
    try:
        conn = get_db_connection()
        
        # Get current activation status
        user = conn.execute('SELECT is_active FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user:
            conn.close()
            flash('User not found', 'error')
            return redirect(url_for('admin_panel'))
        
        # Toggle activation status
        new_status = 1 if user['is_active'] == 0 else 0
        conn.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
        conn.commit()
        
        # Log the action
        action = 'activated' if new_status == 1 else 'deactivated'
        conn.execute('''
            INSERT INTO system_logs (user_id, action, details, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        ''', (session.get('user_id'), f'user_{action}', f'User ID {user_id} {action}'))
        conn.commit()
        conn.close()
        
        flash(f'User {"activated" if new_status == 1 else "deactivated"} successfully', 'success')
        
    except Exception as e:
        flash(f'Error updating user status: {str(e)}', 'error')
    
    return redirect(url_for('admin_panel'))

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@admin_required  
def edit_user(user_id):
    conn = get_db_connection()
    
    if request.method == 'POST':
        try:
            # Get form data
            full_name = request.form.get('full_name')
            badge_id = request.form.get('badge_id')
            department = request.form.get('department')
            rank = request.form.get('rank')
            is_admin = 1 if request.form.get('is_admin') else 0
            
            # Validate required fields
            if not all([full_name, badge_id, department, rank]):
                flash('All fields are required', 'error')
                return redirect(url_for('edit_user', user_id=user_id))
            
            # Check if badge_id is unique (excluding current user)
            existing_user = conn.execute(
                'SELECT id FROM users WHERE badge_id = ? AND id != ?', 
                (badge_id, user_id)
            ).fetchone()
            
            if existing_user:
                flash('Badge ID already exists', 'error')
                return redirect(url_for('edit_user', user_id=user_id))
            
            # Update user
            conn.execute('''
                UPDATE users 
                SET full_name = ?, badge_id = ?, department = ?, rank = ?, is_admin = ?
                WHERE id = ?
            ''', (full_name, badge_id, department, rank, is_admin, user_id))
            conn.commit()
            
            # Log the action
            conn.execute('''
                INSERT INTO system_logs (user_id, action, details, timestamp)
                VALUES (?, ?, ?, datetime('now'))
            ''', (session.get('user_id'), 'user_updated', f'Updated user ID {user_id}'))
            conn.commit()
            
            flash('User updated successfully', 'success')
            return redirect(url_for('view_user', user_id=user_id))
            
        except Exception as e:
            flash(f'Error updating user: {str(e)}', 'error')
        finally:
            conn.close()
    
    # GET request - show edit form
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_panel'))
    
    return render_template('edit_user.html', user=user)

@app.route('/users/<int:user_id>')
@admin_required
def view_user(user_id):
    conn = get_db_connection()
    
    # Get user details
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        flash('User not found', 'error')
        return redirect(url_for('admin_panel'))
    
    # Get user's cases
    user_cases = conn.execute('''
        SELECT case_id, status, matches_found, created_at
        FROM cases 
        WHERE operator_id = ?
        ORDER BY created_at DESC
        LIMIT 10
    ''', (user['badge_id'],)).fetchall()
    
    # Get user's activity logs
    user_logs = conn.execute('''
        SELECT action, details, timestamp
        FROM system_logs 
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 20
    ''', (user_id,)).fetchall()
    
    conn.close()
    
    return render_template('view_user.html', 
                         user=user, 
                         user_cases=user_cases,
                         user_logs=user_logs)


# --- Initialize Database on Startup ---
def create_sample_data():
    """Create some sample data for testing (development only)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if sample data already exists
    existing_cases = cursor.execute('SELECT COUNT(*) as count FROM cases').fetchone()['count']
    
    if existing_cases == 0:
        # Create a few sample cases for demonstration
        sample_cases = [
            ('CASE-2025-001', 'EX001', 'John Smith', 'database', 45, 2, 4.4, 'completed', 'suspect_001.jpg', 12.5),
            ('CASE-2025-002', 'EX001', 'John Smith', 'single', 1, 1, 100.0, 'completed', 'suspect_002.jpg', 3.2),
            ('CASE-2025-003', 'ADMIN001', 'System Administrator', 'database', 67, 0, 0.0, 'completed', 'suspect_003.jpg', 15.8)
        ]
        
        for case_data in sample_cases:
            cursor.execute('''
                INSERT INTO cases (case_id, operator_id, operator_name, analysis_mode, 
                                 total_comparisons, matches_found, match_rate, status, 
                                 suspect_filename, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', case_data)
        
        conn.commit()
        logger.info("Sample data created")
    
    conn.close()

# --- Root Route Redirect ---
@app.route('/')
def index(): 
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Create sample data for development
    create_sample_data()
    
    logger.info("Starting Enhanced Forensic Fingerprint Analysis Web Application")
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Database folder: {app.config['DATABASE_FOLDER']}")
    logger.info(f"Results folder: {app.config['RESULTS_FOLDER']}")
    logger.info("=== DEFAULT CREDENTIALS ===")
    logger.info("Admin - Username: admin, Password: admin123")
    logger.info("Examiner - Username: examiner1, Password: examiner123")
    
    app.run(debug=True, host='0.0.0.0', port=8081)