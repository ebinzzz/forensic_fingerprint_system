"""
Forensic Fingerprint Matching System
====================================
Production-ready fingerprint comparison system for government forensic deployment.
Compliant with forensic standards and includes comprehensive logging and validation.

Author: Forensic IT Division
Version: 1.0.2
Classification: RESTRICTED
"""

import cv2
import numpy as np
import os
import logging
import json
import hashlib
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('forensic_fingerprint.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and other non-serializable objects."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class MatchResult:
    """Structured result for fingerprint matching operations."""
    case_id: str
    image1_hash: str
    image2_hash: str
    match_count: int
    confidence_score: float
    is_match: bool
    threshold_used: float
    processing_time: float
    algorithm_version: str
    operator_id: str
    timestamp: str
    geometric_verification: bool
    quality_score_img1: float
    quality_score_img2: float
    
    def to_serializable_dict(self) -> Dict:
        """Convert to a dictionary with JSON-serializable values."""
        result_dict = asdict(self)
        # Ensure all values are JSON serializable
        for key, value in result_dict.items():
            if isinstance(value, np.bool_):
                result_dict[key] = bool(value)
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                result_dict[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                result_dict[key] = float(value)
        return result_dict
    
class ForensicFingerprintMatcher:
    """
    Production-grade fingerprint matching system for forensic applications.
    
    Features:
    - Multi-stage verification (SIFT + Geometric + Quality assessment)
    - Adaptive thresholding based on image quality
    - Comprehensive audit logging
    - Hash-based evidence integrity
    - Configurable parameters for different deployment scenarios
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the forensic fingerprint matcher."""
        self.version = "1.0.2"
        self.config = self._load_config(config)
        self.sift = cv2.SIFT_create(
            nfeatures=self.config['sift_features'],
            nOctaveLayers=self.config['sift_octave_layers'],
            contrastThreshold=self.config['sift_contrast_threshold']
        )
        
        # Initialize FLANN matcher
        index_params = dict(
            algorithm=self.config['flann_algorithm'],
            trees=self.config['flann_trees']
        )
        search_params = dict(checks=self.config['flann_checks'])
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        logger.info(f"ForensicFingerprintMatcher v{self.version} initialized")
    
    def _load_config(self, config: Optional[Dict] = None) -> Dict:
        """Load configuration with secure defaults."""
        default_config = {
            # SIFT Parameters
            'sift_features': 4000,
            'sift_octave_layers': 4,
            'sift_contrast_threshold': 0.04,
            
            # FLANN Parameters
            'flann_algorithm': 1,   # KD-tree
            'flann_trees': 8,
            'flann_checks': 100,
            
            # Matching Parameters
            'lowe_ratio': 0.7,   # Less strict for better match count
            'min_match_count': 15,
            'geometric_verification': True,
            'adaptive_threshold': True,
            
            # Quality Assessment
            'min_quality_score': 0.2,
            'blur_threshold': 100,
            
            # Security
            'hash_algorithm': 'sha256',
            'audit_logging': True
        }
        
        if config:
            default_config.update(config)
        return default_config
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calculate cryptographic hash of image for integrity verification."""
        try:
            with open(image_path, 'rb') as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {image_path}: {e}")
            return "HASH_ERROR"
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess fingerprint image quality using multiple metrics."""
        try:
            # Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            
            # Contrast measure
            contrast = image.std()
            
            # Ridge clarity (using Sobel gradients)
            sobel_x = cv2.Sobbel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            ridge_clarity = gradient_magnitude.mean()
            
            # Normalize and combine metrics
            quality_score = min(1.0, (laplacian_var / 500 + contrast / 100 + ridge_clarity / 50) / 3)
            return float(quality_score)   # Ensure it's a Python float
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.0
    
    def _enhance_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Advanced fingerprint enhancement for better feature extraction."""
        try:
            # Histogram equalization for contrast enhancement
            enhanced = cv2.equalizeHist(image)
            
            # Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.8)
            
            # Adaptive thresholding for better binarization
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            # Fallback to basic processing
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return binary
    
    def _geometric_verification(self, kp1: List, kp2: List, matches: List) -> Tuple[bool, int]:
        """Perform geometric verification using RANSAC."""
        if len(matches) < 4:   # Minimum points for homography
            return False, 0
        
        try:
            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography using RANSAC
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if homography is None:
                return False, 0
            
            # Count inliers - ensure we get Python int
            inliers = int(np.sum(mask))
            
            # Geometric consistency check
            geometric_valid = (inliers / len(matches)) > 0.4 and inliers >= 8
            
            return bool(geometric_valid), inliers
            
        except Exception as e:
            logger.warning(f"Geometric verification failed: {e}")
            return False, 0
    
    def _calculate_adaptive_threshold(self, img1_quality: float, img2_quality: float, 
                                     total_keypoints: int) -> float:
        """Calculate adaptive threshold based on image quality and keypoint density."""
        if not self.config['adaptive_threshold']:
            return float(self.config['min_match_count'])
        
        base_threshold = self.config['min_match_count']
        
        # Adjust based on average image quality
        avg_quality = (img1_quality + img2_quality) / 2
        quality_factor = 1.0 + (0.5 - avg_quality) * 0.5   # Higher threshold for poor quality
        
        # Adjust based on keypoint density
        density_factor = min(2.0, total_keypoints / 1000)   # Scale with available keypoints
        
        adaptive_threshold = base_threshold * quality_factor * density_factor
        return float(max(8, min(50, adaptive_threshold)))   # Clamp between reasonable bounds
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Return list of supported image formats for forensic analysis."""
        return ['.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.pgm', '.ppm', '.sr', '.ras']
    
    def _validate_image_format(self, image_path: str) -> bool:
        """Validate that image format is supported for forensic analysis."""
        file_ext = Path(image_path).suffix.lower()
        supported = file_ext in self.get_supported_formats()
        
        if not supported:
            logger.warning(f"Unsupported image format: {file_ext}. Supported formats: {self.get_supported_formats()}")
        
        return supported

    def match_fingerprints(self, img1_path: str, img2_path: str, 
                          case_id: str, operator_id: str) -> MatchResult:
        """
        Perform forensic-grade fingerprint matching with comprehensive validation.
        
        Args:
            img1_path: Path to first fingerprint image
            img2_path: Path to second fingerprint image  
            case_id: Unique case identifier for audit trail
            operator_id: ID of the forensic operator
            
        Returns:
            MatchResult object with detailed matching information
        """
        start_time = datetime.now()
        
        try:
            # Validate image formats
            if not self._validate_image_format(img1_path):
                raise ValueError(f"Unsupported format for image 1: {Path(img1_path).suffix}")
            if not self._validate_image_format(img2_path):
                raise ValueError(f"Unsupported format for image 2: {Path(img2_path).suffix}")
            
            # Load and validate images with enhanced error handling
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None:
                raise ValueError(f"Failed to load image 1: {img1_path}. Check file exists and is valid.")
            if img2 is None:
                raise ValueError(f"Failed to load image 2: {img2_path}. Check file exists and is valid.")
            
            # Log image properties for forensic record
            logger.info(f"Case {case_id}: Image 1 - {img1.shape[1]}x{img1.shape[0]} pixels, format: {Path(img1_path).suffix}")
            logger.info(f"Case {case_id}: Image 2 - {img2.shape[1]}x{img2.shape[0]} pixels, format: {Path(img2_path).suffix}")
            
            # Calculate image hashes for integrity
            img1_hash = self._calculate_image_hash(img1_path)
            img2_hash = self._calculate_image_hash(img2_path)
            
            # Assess image quality
            quality1 = self._assess_image_quality(img1)
            quality2 = self._assess_image_quality(img2)
            
            logger.info(f"Case {case_id}: Image quality scores - IMG1: {quality1:.3f}, IMG2: {quality2:.3f}")
            
            # Check minimum quality requirements
            if quality1 < self.config['min_quality_score'] or quality2 < self.config['min_quality_score']:
                logger.warning(f"Case {case_id}: Poor image quality detected")
            
            # Enhance images
            enhanced1 = self._enhance_fingerprint(img1)
            enhanced2 = self._enhance_fingerprint(img2)
            
            # Extract SIFT features
            kp1, des1 = self.sift.detectAndCompute(enhanced1, None)
            kp2, des2 = self.sift.detectAndCompute(enhanced2, None)
            
            if des1 is None or des2 is None:
                logger.warning(f"Case {case_id}: No features detected in one or both images")
                return self._create_no_match_result(case_id, img1_hash, img2_hash, 
                                                     operator_id, start_time, quality1, quality2)
            
            logger.info(f"Case {case_id}: Keypoints detected - IMG1: {len(kp1)}, IMG2: {len(kp2)}")
            
            # Perform FLANN matching
            matches = self.flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.config['lowe_ratio'] * n.distance:
                        good_matches.append(m)
            
            match_count = len(good_matches)
            total_keypoints = len(kp1) + len(kp2)
            
            # Calculate adaptive threshold
            threshold = self._calculate_adaptive_threshold(quality1, quality2, total_keypoints)
            
            # Geometric verification
            geometric_valid = False
            inlier_count = 0
            if self.config['geometric_verification'] and match_count >= 4:
                geometric_valid, inlier_count = self._geometric_verification(kp1, kp2, good_matches)
                logger.info(f"Case {case_id}: Geometric verification - Valid: {geometric_valid}, Inliers: {inlier_count}")
            
            # Calculate confidence score
            confidence = self._calculate_confidence(match_count, total_keypoints, quality1, quality2, 
                                                    geometric_valid, inlier_count)
            
            # Make final decision
            is_match = (match_count >= threshold and 
                        (not self.config['geometric_verification'] or geometric_valid))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = MatchResult(
                case_id=case_id,
                image1_hash=img1_hash,
                image2_hash=img2_hash,
                match_count=int(match_count),
                confidence_score=float(confidence),
                is_match=bool(is_match),
                threshold_used=float(threshold),
                processing_time=float(processing_time),
                algorithm_version=self.version,
                operator_id=operator_id,
                timestamp=datetime.now().isoformat(),
                geometric_verification=bool(geometric_valid),
                quality_score_img1=float(quality1),
                quality_score_img2=float(quality2)
            )
            
            # Audit logging
            if self.config['audit_logging']:
                self._log_match_result(result)
            
            logger.info(f"Case {case_id}: Match decision - {is_match} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Case {case_id}: Matching failed - {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_error_result(case_id, operator_id, str(e), processing_time)
    
    # NEW AND IMPROVED CONFIDENCE SCORE METHOD
    def _calculate_confidence(self, match_count: int, total_keypoints: int,
                              quality1: float, quality2: float,
                              geometric_valid: bool, inlier_count: int) -> float:
        """Calculate a more robust confidence score based on verified matches."""
        
        # Base confidence on the raw number of geometrically verified inliers
        # This is a much stronger signal than total matches.
        if geometric_valid and inlier_count > 0:
            # A logarithmic scale provides a better distribution of scores.
            # This formula ensures confidence grows with the number of inliers.
            base_confidence = np.log1p(inlier_count) / np.log1p(self.config['min_match_count'] + 15)
        else:
            # If geometric verification fails, the confidence should be very low.
            return 0.0

        # Add a minor boost based on image quality.
        avg_quality = (quality1 + quality2) / 2
        quality_factor = 0.8 + 0.2 * avg_quality   # A small boost for high-quality images

        final_confidence = base_confidence * quality_factor

        # Ensure the final score is clamped between 0 and 1.
        return float(min(1.0, max(0.0, final_confidence)))
    
    def _create_no_match_result(self, case_id: str, img1_hash: str, img2_hash: str,
                                operator_id: str, start_time: datetime, 
                                quality1: float, quality2: float) -> MatchResult:
        """Create result for cases with no features detected."""
        processing_time = (datetime.now() - start_time).total_seconds()
        return MatchResult(
            case_id=case_id,
            image1_hash=img1_hash,
            image2_hash=img2_hash,
            match_count=0,
            confidence_score=0.0,
            is_match=False,
            threshold_used=0.0,
            processing_time=float(processing_time),
            algorithm_version=self.version,
            operator_id=operator_id,
            timestamp=datetime.now().isoformat(),
            geometric_verification=False,
            quality_score_img1=float(quality1),
            quality_score_img2=float(quality2)
        )
    
    def _create_error_result(self, case_id: str, operator_id: str, 
                             error_msg: str, processing_time: float) -> MatchResult:
        """Create result for error cases."""
        return MatchResult(
            case_id=case_id,
            image1_hash="ERROR",
            image2_hash="ERROR",
            match_count=-1,
            confidence_score=0.0,
            is_match=False,
            threshold_used=0.0,
            processing_time=float(processing_time),
            algorithm_version=self.version,
            operator_id=operator_id,
            timestamp=datetime.now().isoformat(),
            geometric_verification=False,
            quality_score_img1=0.0,
            quality_score_img2=0.0
        )
    
    def _log_match_result(self, result: MatchResult) -> None:
        """Log match result for audit trail."""
        audit_entry = {
            'event_type': 'FINGERPRINT_MATCH',
            'result': result.to_serializable_dict(),
            'system_info': {
                'opencv_version': cv2.__version__,
                'algorithm': 'SIFT+FLANN+Geometric'
            }
        }
        # Use custom encoder to handle any remaining numpy types
        logger.info(f"AUDIT: {json.dumps(audit_entry, cls=CustomJSONEncoder)}")
    
    def process_case_batch(self, cases: List[Dict], results_folder: str, 
                          operator_id: str) -> Dict[str, Any]:
        """
        Process a batch of fingerprint comparison cases.
        
        Args:
            cases: List of case dictionaries with 'case_id', 'img1_path', 'img2_path', 'ground_truth'
            results_folder: Folder to save results
            operator_id: ID of the forensic operator
            
        Returns:
            Dictionary with batch processing results and performance metrics
        """
        results_folder = Path(results_folder)
        results_folder.mkdir(exist_ok=True)
        
        results = []
        y_true = []
        y_pred = []
        confidences = []
        
        logger.info(f"Processing batch of {len(cases)} cases")
        
        for case in cases:
            try:
                result = self.match_fingerprints(
                    case['img1_path'], 
                    case['img2_path'],
                    case['case_id'],
                    operator_id
                )
                
                results.append(result)
                y_true.append(case.get('ground_truth', 0))
                y_pred.append(int(result.is_match))
                confidences.append(result.confidence_score)
                
                # Save individual result using the serializable dictionary
                result_file = results_folder / f"{case['case_id']}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result.to_serializable_dict(), f, indent=2, cls=CustomJSONEncoder)
                
            except Exception as e:
                logger.error(f"Failed to process case {case.get('case_id', 'UNKNOWN')}: {e}")
        
        # Generate performance report
        performance_report = self._generate_performance_report(
            y_true, y_pred, confidences, results_folder
        )
        
        batch_summary = {
            'total_cases': len(cases),
            'processed_cases': len(results),
            'performance_metrics': performance_report,
            'results': [r.to_serializable_dict() for r in results]
        }
        
        # Save batch summary
        summary_file = results_folder / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, cls=CustomJSONEncoder)
        
        logger.info(f"Batch processing complete. Summary saved to {summary_file}")
        return batch_summary
    
    def _generate_performance_report(self, y_true: List, y_pred: List, 
                                     confidences: List, results_folder: Path) -> Dict:
        """Generate comprehensive performance metrics."""
        try:
            if not y_true or not y_pred:
                logger.warning("No ground truth data available for performance evaluation")
                return {'error': 'No ground truth data available'}
            
            # Basic metrics
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # Handle different confusion matrix shapes
            unique_labels = sorted(list(set(y_true + y_pred)))
            
            if len(unique_labels) == 1:
                # Only one class present
                if unique_labels[0] == 1:
                    tp, fp, fn, tn = cm[0, 0], 0, 0, 0
                else:
                    tp, fp, fn, tn = 0, 0, 0, cm[0, 0]
            elif len(unique_labels) == 2:
                # Both classes present
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    # Handle edge case
                    tn = fp = fn = tp = 0
                    for i, true_label in enumerate(unique_labels):
                        for j, pred_label in enumerate(unique_labels):
                            if true_label == 0 and pred_label == 0:
                                tn = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
                            elif true_label == 0 and pred_label == 1:
                                fp = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
                            elif true_label == 1 and pred_label == 0:
                                fn = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
                            elif true_label == 1 and pred_label == 1:
                                tp = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
            else:
                tp = fp = fn = tn = 0
            
            metrics = {
                'accuracy': float(report['accuracy']),
                'precision': float(report.get('1', {}).get('precision', 0)),
                'recall': float(report.get('1', {}).get('recall', 0)),
                'f1_score': float(report.get('1', {}).get('f1-score', 0)),
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'unique_labels': unique_labels,
                'confusion_matrix_shape': cm.shape
            }
            
            # Generate confusion matrix plot - handle single class case
            try:
                plt.figure(figsize=(8, 6))
                
                if len(unique_labels) == 1:
                    # Single class case - create a simple display
                    label_name = "Same" if unique_labels[0] == 1 else "Different"
                    plt.imshow([[cm[0, 0]]], cmap="Blues", vmin=0)
                    plt.colorbar()
                    plt.text(0, 0, str(cm[0, 0]), ha='center', va='center', fontsize=14, fontweight='bold')
                    plt.xticks([0], [label_name])
                    plt.yticks([0], [label_name])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f"Forensic Fingerprint Matching - Results (All {label_name})")
                else:
                    # Standard confusion matrix for multiple classes
                    display_labels = []
                    for label in sorted(unique_labels):
                        display_labels.append("Same" if label == 1 else "Different")
                    
                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm, 
                        display_labels=display_labels
                    )
                    disp.plot(cmap="Blues", values_format="d")
                    plt.title("Forensic Fingerprint Matching - Confusion Matrix")
                
                plt.tight_layout()
                
                confusion_matrix_path = results_folder / "confusion_matrix.png"
                plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Confusion matrix saved to {confusion_matrix_path}")
                
            except Exception as plot_error:
                logger.warning(f"Confusion matrix plot generation failed: {plot_error}")
                plt.close()   # Ensure plot is closed even if error occurs
            
            logger.info(f"Performance report generated: Accuracy={metrics['accuracy']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}