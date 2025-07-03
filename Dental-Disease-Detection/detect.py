import argparse
from ultralytics import YOLO
import os
import json
from datetime import datetime

# Class names from data.yaml
CLASS_NAMES = ['Caries', 'Ulcer', 'Tooth Discoloration', 'Gingivitis']

# Suggestions for each class
suggestions = {
    'Caries': [
        "Schedule a dental filling immediately",
        "Use fluoride toothpaste",
        "Limit sugary snacks and drinks",
        "Consider dental sealants"
    ],
    'Ulcer': [
        "Use topical oral gels for pain relief",
        "Avoid spicy/acidic foods",
        "Maintain oral hygiene",
        "Monitor healing progress"
    ],
    'Tooth Discoloration': [
        "Consult a dentist for whitening options",
        "Avoid staining beverages like coffee/tea",
        "Maintain proper brushing habits",
        "Consider professional cleaning"
    ],
    'Gingivitis': [
        "Floss daily to remove plaque",
        "Use therapeutic mouthwash",
        "Schedule a professional dental cleaning",
        "Maintain proper brushing technique"
    ]
}

def generate_report(results, output_dir):
    """Generate a comprehensive report with suggestions"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "findings": [],
        "summary": {
            "total_findings": 0,
            "urgent_cases": 0
        }
    }

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = CLASS_NAMES[cls]

            finding = {
                "condition": class_name,
                "confidence": conf,
                "suggestions": suggestions.get(class_name, []),
                "location": box.xywh.tolist()
            }

            report["findings"].append(finding)
            report["summary"]["total_findings"] += 1
            if class_name in ['Caries', 'Gingivitis']:
                report["summary"]["urgent_cases"] += 1

    # Save reports
    json_path = os.path.join(output_dir, "dental_report.json")
    txt_path = os.path.join(output_dir, "recommendations.txt")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("Dental Health Recommendations\n")
        f.write("=" * 40 + "\n")
        for finding in report["findings"]:
            f.write(f"\nCondition: {finding['condition']} ({finding['confidence']:.2%})\n")
            f.write("Suggestions:\n")
            for suggestion in finding['suggestions']:
                f.write(f"- {suggestion}\n")

    return json_path, txt_path


def main(args):
    # Validate inputs
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source path {args.source} does not exist")

    # Setup output
    output_dir = args.output_dir or os.path.dirname(args.source)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = YOLO(args.weights)

    # Run inference
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save
    )

    # Generate reports
    json_report, txt_report = generate_report(results, output_dir)

    print(f"\nAnalysis Complete!")
    print(f"Total Findings: {len(results[0].boxes)}")
    print(f"JSON Report: {json_report}")
    print(f"Recommendations: {txt_report}")
    if args.save:
        print(f"Visual Results: {results[0].save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dental Health Analysis CLI")
    parser.add_argument("--weights", type=str, required=True, help="Model weights path")
    parser.add_argument("--source", type=str, required=True, help="Input image/directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--save", action="store_true", help="Save detection visuals")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
