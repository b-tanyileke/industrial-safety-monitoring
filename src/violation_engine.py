"""
This file defines the ViolationEngine class, which is responsible 
for evaluating detected persons and their PPE against defined safety rules.
The engine checks for violations such as not wearing a helmet, not wearing a vest, and 
being in a restricted zone. It returns a list of violations for each detected person.
"""

class ViolationEngine:
    """Evaluates detected persons and their PPE against defined safety rules to identify violations.
    """

    def evaluate(self, detections, zone_manager):
        """Evaluates the detections against safety rules and identifies violations.
        Args:
            detections (list): A list of dictionaries containing detection information for each person.
            zone_manager (ZoneManager): An instance of the ZoneManager class to check zone violations.
        Returns:
            list: A list of dictionaries containing violation information for each detected person."""

        violations = []

        for idx, detection in enumerate(detections):

            # Extract the person's bounding box and PPE classes
            person_box = detection["person_box"]

            ppe_classes = [
                item["class"]
                for item in detection["ppe"]
            ]

            if "no helmet" in ppe_classes or "helmet" not in ppe_classes:
                # Log a violation for not wearing a helmet
                violations.append({
                    "person_id": detection.get("person_id", idx),
                    "type": "No Helmet"
                })

            if "no vest" in ppe_classes or "vest" not in ppe_classes:

                violations.append({
                    "person_id": detection.get("person_id", idx),
                    "type": "No Vest"
                })

            if zone_manager.is_inside_zone(person_box):

                violations.append({
                    "person_id": detection.get("person_id", idx),
                    "type": "Restricted Zone"
                })

        return violations
    