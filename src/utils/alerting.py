import time
import os

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from utils.config import load_config

def map_severity(max_width_mm, length_mm, severity_thresholds=None):
    """
    Maps crack width and length to API 570/579 severity levels.
    """
    if severity_thresholds is None:
        try:
            config = load_config()
            severity_thresholds = config.get("alerting", {}).get("severity_thresholds", {})
        except Exception:
            severity_thresholds = {}
            
    # Default values in case they are missing in config
    l3 = severity_thresholds.get("level_3", {"max_width_mm": 0.5, "length_mm": 50.0, "status": "CRITICAL", "recommended_action": "Immediate shutdown & emergency maintenance inspection"})
    l2 = severity_thresholds.get("level_2", {"max_width_mm": 0.2, "length_mm": 20.0, "status": "MODERATE", "recommended_action": "Schedule repair & maintenance within 30 days"})
    l1 = severity_thresholds.get("level_1", {"status": "MINOR", "recommended_action": "Routine monitoring and logging during next service cycle"})

    if max_width_mm > l3.get("max_width_mm", 0.5) or length_mm > l3.get("length_mm", 50.0):
        return {
            "level": 3,
            "status": l3.get("status", "CRITICAL"),
            "recommended_action": l3.get("recommended_action", "Immediate shutdown & emergency maintenance inspection")
        }
    elif max_width_mm > l2.get("max_width_mm", 0.2) or length_mm > l2.get("length_mm", 20.0):
        return {
            "level": 2,
            "status": l2.get("status", "MODERATE"),
            "recommended_action": l2.get("recommended_action", "Schedule repair & maintenance within 30 days")
        }
    else:
        return {
            "level": 1,
            "status": l1.get("status", "MINOR"),
            "recommended_action": l1.get("recommended_action", "Routine monitoring and logging during next service cycle")
        }

class AlertSystem:
    def __init__(self, log_path="alerts.log", config=None):
        self.log_path = log_path
        
        # Load config if not provided
        if config is None:
            try:
                global_config = load_config()
                config = global_config.get("alerting", {})
            except Exception:
                config = {}
                
        # Parse cooldowns
        cooldown_config = config.get("cooldowns", {})
        self.cooldowns = {}
        for k, v in cooldown_config.items():
            try:
                self.cooldowns[int(k)] = float(v)
            except ValueError:
                pass
                
        # Provide defaults if empty
        if not self.cooldowns:
            self.cooldowns = {2: 7200, 3: 600}
            
        self.last_alert_time = {k: 0.0 for k in self.cooldowns.keys()}
        self.severity_thresholds = config.get("severity_thresholds", None)

        
    def log_alert(self, severity_info, frame_id, max_width_mm, length_mm):
        level = severity_info["level"]
        status = severity_info["status"]
        action = severity_info["recommended_action"]
        
        log_line = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"ALERT Level {level} ({status}) - Frame ID: {frame_id} - "
            f"Max Width: {max_width_mm:.2f}mm, Length: {length_mm:.1f}mm. "
            f"Action Required: {action}\n"
        )
        
        print(log_line.strip())
        with open(self.log_path, "a") as f:
            f.write(log_line)
            
    def trigger_alert(self, severity_info, frame_id, max_width_mm, length_mm):
        level = severity_info["level"]
        current_time = time.time()
        
        # Log to MLflow if available
        if HAS_MLFLOW and mlflow.active_run() is not None:
            mlflow.log_metric(f"severity_level", level, step=int(frame_id))
            mlflow.log_metric(f"max_width_mm", max_width_mm, step=int(frame_id))
            mlflow.log_metric(f"length_mm", length_mm, step=int(frame_id))
            
        # Minor severity alerts are just logged locally and to MLflow, no external push/alert
        if level == 1:
            self.log_alert(severity_info, frame_id, max_width_mm, length_mm)
            return True
            
        # Check cooldown for Level 2 and Level 3
        cooldown_period = self.cooldowns.get(level, 0.0)
        time_elapsed = current_time - self.last_alert_time.get(level, 0.0)
        
        if time_elapsed >= cooldown_period:
            self.last_alert_time[level] = current_time
            self.log_alert(severity_info, frame_id, max_width_mm, length_mm)
            # (Here is where we would hook into Slack, Twilio SMS, or Sendgrid email API)
            print(f">>> Slack/SMS Alert Sent for Level {level}! (Cooldown active for {cooldown_period}s)")
            return True
        else:
            remaining = cooldown_period - time_elapsed
            print(f"[Cooldown] Suppressed alert for Level {level}. Re-alerting in {remaining:.1f}s")
            return False
