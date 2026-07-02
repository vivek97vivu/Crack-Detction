import time
import os

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

def map_severity(max_width_mm, length_mm):
    """
    Maps crack width and length to API 570/579 severity levels.
    """
    if max_width_mm > 0.5 or length_mm > 50.0:
        return {
            "level": 3,
            "status": "CRITICAL",
            "recommended_action": "Immediate shutdown & emergency maintenance inspection"
        }
    elif max_width_mm > 0.2 or length_mm > 20.0:
        return {
            "level": 2,
            "status": "MODERATE",
            "recommended_action": "Schedule repair & maintenance within 30 days"
        }
    else:
        return {
            "level": 1,
            "status": "MINOR",
            "recommended_action": "Routine monitoring and logging during next service cycle"
        }

class AlertSystem:
    def __init__(self, log_path="alerts.log"):
        self.log_path = log_path
        # Cooldown trackers store the last time an alert was fired for each level
        # Cooldown intervals in seconds: Level 2 -> 2 hours (7200s), Level 3 -> 10 minutes (600s)
        self.cooldowns = {2: 7200, 3: 600}
        self.last_alert_time = {2: 0.0, 3: 0.0}
        
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
