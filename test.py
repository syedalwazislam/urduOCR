"""
test_cnic_extraction.py
------------------------
Test script for CNIC parser with bounding boxes and regex CNIC extraction.
"""

import re
import cv2
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

# Import your CNIC processor
from cnic_parser import CNICProcessor, EnhancedCNICProcessor

class CNICRegexTester:
    """Test class for CNIC extraction with regex."""
    
    def __init__(self, ocr_engine: str = "easyocr"):
        """
        Initialize tester.
        
        Parameters
        ----------
        ocr_engine : str
            'paddle', 'easyocr', or 'tesseract'
        """
        self.ocr_engine = ocr_engine
        self.processor = EnhancedCNICProcessor(ocr_engine=ocr_engine, use_bounding_boxes=True)
        
        # CNIC regex patterns
        self.cnic_patterns = [
            re.compile(r'\b\d{5}-\d{7}-\d{1}\b'),      # Standard format: 12345-1234567-1
            re.compile(r'\b\d{13}\b'),                 # Without hyphens: 1234512345671
            re.compile(r'\d{5}\s*\d{7}\s*\d{1}'),      # With spaces: 12345 1234567 1
        ]
        
        # Date regex patterns
        self.date_patterns = [
            re.compile(r'\b(\d{2})[\/\-](\d{2})[\/\-](\d{4})\b'),  # DD/MM/YYYY or DD-MM-YYYY
            re.compile(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b'), # D/M/YYYY
        ]
    
    def extract_cnic_with_regex(self, text: str) -> Optional[str]:
        """
        Extract CNIC number using regex patterns.
        
        Parameters
        ----------
        text : str
            Extracted text from OCR
            
        Returns
        -------
        str or None
            Formatted CNIC number
        """
        if not text:
            return None
        
        # Try each pattern
        for pattern in self.cnic_patterns:
            match = pattern.search(text)
            if match:
                cnic = match.group(0)
                # Format if needed
                if re.match(r'\d{13}', cnic):
                    # Format without hyphens to with hyphens
                    return f"{cnic[:5]}-{cnic[5:12]}-{cnic[12]}"
                elif re.match(r'\d{5}\s*\d{7}\s*\d{1}', cnic):
                    # Clean spaces and add hyphens
                    clean = re.sub(r'\s', '', cnic)
                    return f"{clean[:5]}-{clean[5:12]}-{clean[12]}"
                return cnic
        
        return None
    
    def extract_all_cnics(self, text: str) -> list:
        """Extract all CNIC numbers from text."""
        cnics = []
        for pattern in self.cnic_patterns:
            matches = pattern.findall(text)
            for match in matches:
                clean_match = re.sub(r'\s', '', match)
                if len(clean_match) == 13:
                    formatted = f"{clean_match[:5]}-{clean_match[5:12]}-{clean_match[12]}"
                    cnics.append(formatted)
                else:
                    cnics.append(match)
        return list(set(cnics))  # Remove duplicates
    
    def extract_date_with_regex(self, text: str) -> Optional[str]:
        """
        Extract date using regex patterns.
        
        Parameters
        ----------
        text : str
            Extracted text from OCR
            
        Returns
        -------
        str or None
            Extracted date
        """
        if not text:
            return None
        
        for pattern in self.date_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        
        return None
    
    def test_single_cnic(self, front_path: str, back_path: str) -> Dict:
        """
        Test CNIC extraction on single pair of images.
        
        Parameters
        ----------
        front_path : str
            Path to front image
        back_path : str
            Path to back image
            
        Returns
        -------
        dict
            Test results
        """
        print(f"\n{'='*60}")
        print(f"Testing CNIC Extraction")
        print(f"Front: {Path(front_path).name}")
        print(f"Back:  {Path(back_path).name}")
        print(f"{'='*60}")
        
        results = {
            "front_image": front_path,
            "back_image": back_path,
            "extracted": {},
            "regex_extracted": {},
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Process with bounding boxes
            extraction_result = self.processor.process_cnic_with_boxes(front_path, back_path)
            results["extracted"] = extraction_result
            
            # Apply regex on extracted text fields
            for side in ["front", "back"]:
                for field, value in extraction_result.get(side, {}).items():
                    if value and isinstance(value, str):
                        # Extract CNIC using regex
                        if "cnic" in field or "shanakhti" in field:
                            cnic = self.extract_cnic_with_regex(value)
                            if cnic:
                                results["regex_extracted"][f"{side}_{field}"] = cnic
                        
                        # Extract dates using regex
                        if "date" in field or "tareekh" in field:
                            date = self.extract_date_with_regex(value)
                            if date:
                                results["regex_extracted"][f"{side}_{field}"] = date
            
            # Also search entire extracted text for CNIC numbers
            all_text = ""
            for side in ["front", "back"]:
                for field, value in extraction_result.get(side, {}).items():
                    if value and isinstance(value, str):
                        all_text += value + " "
            
            all_cnics = self.extract_all_cnics(all_text)
            if all_cnics:
                results["regex_extracted"]["all_cnics_found"] = all_cnics
                # Take first CNIC as primary
                if "back_shanakhti_number" not in results["regex_extracted"]:
                    results["regex_extracted"]["primary_cnic"] = all_cnics[0]
            
            results["processing_time"] = time.time() - start_time
            
            # Display results
            self._display_results(results)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
        
        return results
    
    def _display_results(self, results: Dict):
        """Display test results."""
        print("\n📋 EXTRACTION RESULTS:")
        print("-" * 40)
        
        # Display bounding box extraction
        extracted = results.get("extracted", {})
        
        print("\n🔹 Front Side (Bounding Box):")
        for field, value in extracted.get("front", {}).items():
            print(f"   {field}: {value if value else '❌ Not found'}")
        
        print("\n🔸 Back Side (Bounding Box):")
        for field, value in extracted.get("back", {}).items():
            print(f"   {field}: {value if value else '❌ Not found'}")
        
        # Display regex extraction results
        print("\n🎯 Regex-Based CNIC Extraction:")
        regex_extracted = results.get("regex_extracted", {})
        
        if regex_extracted:
            for field, value in regex_extracted.items():
                if "cnic" in field.lower() or "shanakhti" in field.lower():
                    print(f"   ✅ CNIC Found ({field}): {value}")
            
            if "primary_cnic" in regex_extracted:
                print(f"\n   🆔 Primary CNIC: {regex_extracted['primary_cnic']}")
        else:
            print("   ❌ No CNIC found via regex")
        
        print(f"\n⏱ Processing Time: {results.get('processing_time', 0):.2f} seconds")
    
    def batch_test(self, test_cases: list) -> Dict:
        """
        Run batch tests on multiple CNIC images.
        
        Parameters
        ----------
        test_cases : list
            List of tuples (front_path, back_path, expected_cnic)
            
        Returns
        -------
        dict
            Batch test summary
        """
        print(f"\n{'#'*60}")
        print(f"BATCH TEST - {len(test_cases)} CNIC(s)")
        print(f"{'#'*60}")
        
        all_results = []
        successful_extractions = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📌 Test Case {i}/{len(test_cases)}")
            
            if len(test_case) == 3:
                front, back, expected_cnic = test_case
            else:
                front, back = test_case
                expected_cnic = None
            
            result = self.test_single_cnic(front, back)
            result["expected_cnic"] = expected_cnic
            
            # Validate against expected
            if expected_cnic:
                extracted_cnic = result.get("regex_extracted", {}).get("primary_cnic")
                if extracted_cnic == expected_cnic:
                    successful_extractions += 1
                    result["validation"] = "✅ PASS"
                else:
                    result["validation"] = f"❌ FAIL (Expected: {expected_cnic})"
            
            all_results.append(result)
        
        # Summary
        print(f"\n{'='*60}")
        print("BATCH TEST SUMMARY")
        print(f"{'='*60}")
        
        total = len(all_results)
        passed = sum(1 for r in all_results if r.get("validation") == "✅ PASS")
        
        print(f"\n📊 Statistics:")
        print(f"   Total Tests: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {total - passed}")
        print(f"   Success Rate: {(passed/total)*100:.1f}%")
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"cnic_test_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed/total)*100 if total > 0 else 0,
            "results": all_results
        }

def visualize_bounding_boxes_for_test(front_path: str, back_path: str):
    """
    Visualize bounding boxes for debugging.
    
    Parameters
    ----------
    front_path : str
        Path to front image
    back_path : str
        Path to back image
    """
    from cnic_parser import CNICBoundingBoxExtractor
    
    print("\n🎨 Visualizing Bounding Boxes...")
    
    # Front side
    front_extractor = CNICBoundingBoxExtractor(front_path)
    front_output = front_extractor.visualize_bounding_boxes("front_bboxes_debug.jpg")
    print(f"   ✓ Front boxes saved to: {front_output}")
    
    # Back side
    back_extractor = CNICBoundingBoxExtractor(back_path)
    back_output = back_extractor.visualize_bounding_boxes("back_bboxes_debug.jpg")
    print(f"   ✓ Back boxes saved to: {back_output}")

def test_regex_only(text_samples: list):
    """
    Test regex extraction on text samples.
    
    Parameters
    ----------
    text_samples : list
        List of text strings to test regex on
    """
    print("\n🔍 Testing Regex Extraction on Text Samples")
    print("=" * 60)
    
    tester = CNICRegexTester()
    
    for i, text in enumerate(text_samples, 1):
        print(f"\n📝 Sample {i}:")
        print(f"   Text: {text}")
        
        cnic = tester.extract_cnic_with_regex(text)
        print(f"   Extracted CNIC: {cnic if cnic else '❌ Not found'}")
        
        cnics = tester.extract_all_cnics(text)
        if cnics:
            print(f"   All CNICs found: {cnics}")

def interactive_test():
    """Interactive test mode."""
    print("\n🔍 CNIC Extraction Interactive Test")
    print("=" * 50)
    
    # Choose OCR engine
    print("\nOCR Engine Options:")
    print("1. EasyOCR (Recommended)")
    print("2. PaddleOCR")
    print("3. Tesseract")
    
    choice = input("\nSelect engine (1/2/3): ").strip()
    engine_map = {"1": "easyocr", "2": "paddle", "3": "tesseract"}
    engine = engine_map.get(choice, "easyocr")
    
    # Initialize tester
    tester = CNICRegexTester(ocr_engine=engine)
    
    while True:
        print("\n" + "-" * 40)
        front_path = input("Enter front image path (or 'quit'): ").strip()
        
        if front_path.lower() == 'quit':
            break
        
        if not Path(front_path).exists():
            print(f"❌ File not found: {front_path}")
            continue
        
        back_path = input("Enter back image path: ").strip()
        
        if not Path(back_path).exists():
            print(f"❌ File not found: {back_path}")
            continue
        
        # Show bounding boxes first
        visualize = input("Show bounding box visualization? (y/n): ").strip().lower()
        if visualize == 'y':
            visualize_bounding_boxes_for_test(front_path, back_path)
        
        # Test extraction
        results = tester.test_single_cnic(front_path, back_path)
        
        # Ask for expected CNIC
        expected = input("\nEnter expected CNIC (or press Enter to skip): ").strip()
        if expected:
            extracted = results.get("regex_extracted", {}).get("primary_cnic")
            if extracted == expected:
                print("✅ Validation: MATCH!")
            else:
                print(f"❌ Validation: MISMATCH (Expected: {expected}, Got: {extracted})")

def main():
    """Main test function."""
    print("CNIC Extraction Test Suite")
    print("=" * 60)
    
    # Test regex patterns on sample text
    test_samples = [
        "15601-6009035-1",
        "CNIC: 12345-1234567-8",
        "شناختی نمبر: 42101-1234567-8",
        "1560160090351",
        "12345 1234567 1",
    ]
    
    test_regex_only(test_samples)
    
    # Ask for image testing
    print("\n" + "="*60)
    run_image_test = input("\nTest with actual images? (y/n): ").strip().lower()
    
    if run_image_test == 'y':
        # Check if we have test images
        test_front = Path("test_front.jpg")
        test_back = Path("test_back.jpg")
        
        if test_front.exists() and test_back.exists():
            # Run test on existing images
            tester = CNICRegexTester(ocr_engine="easyocr")
            visualize_bounding_boxes_for_test(str(test_front), str(test_back))
            results = tester.test_single_cnic(str(test_front), str(test_back))
            
            # Try to extract CNIC from results
            print("\n🎯 Final Extracted CNIC:", results.get("regex_extracted", {}).get("primary_cnic", "Not found"))
        else:
            # Interactive mode
            interactive_test()
    
    # Batch test example
    print("\n" + "="*60)
    run_batch = input("\nRun batch test with sample data? (y/n): ").strip().lower()
    
    if run_batch == 'y':
        # Create sample test cases
        sample_cases = []
        
        # Check for existing test images
        if test_front.exists() and test_back.exists():
            sample_cases.append((str(test_front), str(test_back), "15601-6009035-1"))
        
        if sample_cases:
            tester = CNICRegexTester(ocr_engine="easyocr")
            tester.batch_test(sample_cases)
        else:
            print("No test images found. Please add test_front.jpg and test_back.jpg")

if __name__ == "__main__":
    main()