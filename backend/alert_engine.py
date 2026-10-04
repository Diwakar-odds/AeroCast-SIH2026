"""
Common Alerting Protocol (CAP v1.2) Engine
Translates severe weather predictions into OASIS & NDMA compliant XML/JSON alerts
"""
import datetime
import uuid
import xml.etree.ElementTree as ET

class CAPAlertEngine:
    def __init__(self, sender_id: str = "alerts@aerocast.ncmrwf.gov.in"):
        self.sender_id = sender_id

    def generate_cap_alert(self, event_type: str, severity: str, region: str, coords: str, description: str):
        alert_id = f"AEROCAST-{uuid.uuid4().hex[:8].upper()}"
        sent_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Root XML element
        alert = ET.Element("alert", xmlns="urn:oasis:names:tc:emergency:cap:1.2")
        ET.SubElement(alert, "identifier").text = alert_id
        ET.SubElement(alert, "sender").text = self.sender_id
        ET.SubElement(alert, "sent").text = sent_time
        ET.SubElement(alert, "status").text = "Actual"
        ET.SubElement(alert, "msgType").text = "Alert"
        ET.SubElement(alert, "scope").text = "Public"

        info = ET.SubElement(alert, "info")
        ET.SubElement(info, "category").text = "Met"
        ET.SubElement(info, "event").text = event_type
        ET.SubElement(info, "urgency").text = "Immediate"
        ET.SubElement(info, "severity").text = severity
        ET.SubElement(info, "certainty").text = "Observed"
        ET.SubElement(info, "headline").text = f"{severity.upper()} WARNING: {event_type} in {region}"
        ET.SubElement(info, "description").text = description

        area = ET.SubElement(info, "area")
        ET.SubElement(area, "areaDesc").text = region
        ET.SubElement(area, "circle").text = coords

        xml_str = ET.tostring(alert, encoding="utf-8", method="xml").decode("utf-8")
        
        json_payload = {
            "identifier": alert_id,
            "sender": self.sender_id,
            "sent": sent_time,
            "status": "Actual",
            "msgType": "Alert",
            "info": {
                "event": event_type,
                "severity": severity,
                "region": region,
                "coordinates": coords,
                "description": description
            }
        }
        return xml_str, json_payload

if __name__ == "__main__":
    engine = CAPAlertEngine()
    xml, js = engine.generate_cap_alert(
        "Cloudburst & Flash Flood",
        "Extreme",
        "Mandi-Kullu Catchment, Himachal Pradesh",
        "31.7087,76.9320,25.0",
        "Extreme convective storm approaching. Rain rates exceeding 100mm/hr projected within 2.5 hours."
    )
    print("CAP Alert Generated Successfully:")
    print(xml[:300] + "...")
