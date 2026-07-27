import smtplib
import json
import time
import os
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# Carrier email-to-SMS gateways (number@carrier_gateway)
CARRIER_GATEWAYS = {
    "att": "@txt.att.net",
    "verizon": "@vtext.com",
    "tmobile": "@tmomail.net",
    "sprint": "@messaging.sprintpcs.com",
    "boost": "@myboostmobile.com",
    "cricket": "@mms.cricketwireless.net",
    "google": "@msg.fi.google.com",
    "uscellular": "@email.uscc.net",
    "metro": "@mymetropcs.com",
}


class AlertManager:
    """
    Multi-channel alert system for security events.

    Supports:
    - Email via SMTP (Gmail, Outlook, custom servers)
    - Telegram Bot notifications
    - Discord Webhook embeds
    - SMS via carrier email-to-SMS gateways
    - Generic webhooks (IFTTT, Zapier, custom endpoints)

    Features:
    - Per-event cooldown to prevent alert spam
    - Severity-based routing (CRITICAL goes to all channels)
    - Configurable dangerous object list
    - Alert history logging
    """

    # Events that always trigger alerts regardless of confidence
    CRITICAL_EVENTS = {
        "SPOOF_DETECTED", "EVACUATION_ALERT", "CROWD_FORMING",
        "DANGEROUS_OBJECT",
    }

    # Events that trigger alerts at WARNING level
    WARNING_EVENTS = {
        "HESITATION", "PACING", "SCANNING", "SPATIAL_ANOMALY",
        "LOITERING", "RUNNING", "OBJECT_INTERACTION",
        "CONGESTION",
    }

    # Dangerous objects to watch for (from YOLO class names, lowercase)
    DANGEROUS_OBJECTS = {
        "knife", "scissors", "gun", "pistol", "rifle", "weapon",
        "bat", "axe", "hammer", "baseball bat", "cleaver",
        "machete", "sword", "explosive", "bomb", "blade",
    }

    def __init__(self, config_path=None, config=None):
        """
        Initialize AlertManager.

        Args:
            config_path: Path to a JSON config file with alert settings.
            config: Direct dict of alert settings (overrides file config).
        """
        self.config = {}

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                print(f"[ALERT] Loaded config from {config_path}")
            except Exception as e:
                print(f"[ALERT WARN] Failed to load config: {e}")

        # Override with direct config
        if config:
            self.config.update(config)

        # Cooldown settings
        # FIX #3: Both __init__ and create_sample_config now use "critical_cooldown"
        self.alert_cooldown_sec = self.config.get("alert_cooldown_sec", 300)
        self.critical_cooldown_sec = self.config.get("critical_cooldown", 60)
        self.last_alert_time = {}   # {alert_key: timestamp}
        self.alert_history = []     # list of dicts for logging

        # Email settings
        email_cfg = self.config.get("email", {})
        self.email_enabled = email_cfg.get("enabled", False)
        self.smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = email_cfg.get("smtp_port", 587)
        self.smtp_user = email_cfg.get("smtp_user", "")
        self.smtp_pass = email_cfg.get("smtp_pass", "")
        self.email_from = email_cfg.get("from", self.smtp_user)
        self.email_to = email_cfg.get("to", [])
        if isinstance(self.email_to, str):
            self.email_to = [self.email_to]

        # Telegram settings
        telegram_cfg = self.config.get("telegram", {})
        self.telegram_enabled = telegram_cfg.get("enabled", False)
        self.telegram_token = telegram_cfg.get("bot_token", "")
        self.telegram_chat_id = telegram_cfg.get("chat_id", "")

        # Discord settings
        discord_cfg = self.config.get("discord", {})
        self.discord_enabled = discord_cfg.get("enabled", False)
        self.discord_webhook_url = discord_cfg.get("webhook_url", "")

        # SMS settings (email-to-SMS)
        sms_cfg = self.config.get("sms", {})
        self.sms_enabled = sms_cfg.get("enabled", False)
        self.sms_recipients = sms_cfg.get("recipients", [])
        # Each recipient: {"number": "1234567890", "carrier": "att"}

        # Generic webhook
        webhook_cfg = self.config.get("webhook", {})
        self.webhook_enabled = webhook_cfg.get("enabled", False)
        self.webhook_url = webhook_cfg.get("url", "")
        self.webhook_method = webhook_cfg.get("method", "POST")
        self.webhook_headers = webhook_cfg.get("headers", {"Content-Type": "application/json"})

        # Global enable/disable
        self.enabled = self.config.get("enabled", True)

        # Validate
        self._validate_config()

        # Log init
        channels = []
        if self.email_enabled:
            channels.append(f"Email({self.smtp_server})")
        if self.telegram_enabled:
            channels.append("Telegram")
        if self.discord_enabled:
            channels.append("Discord")
        if self.sms_enabled:
            channels.append(f"SMS({len(self.sms_recipients)} recipients)")
        if self.webhook_enabled:
            channels.append("Webhook")

        if channels:
            print(f"[ALERT] Alert system active. Channels: {', '.join(channels)}")
        else:
            print("[ALERT] WARNING: No alert channels configured! Alerts will only print to console.")

    def _validate_config(self):
        """Basic validation of config."""
        if self.email_enabled and not self.smtp_user:
            print("[ALERT WARN] Email enabled but no SMTP username configured.")
        if self.telegram_enabled and not self.telegram_token:
            print("[ALERT WARN] Telegram enabled but no bot token configured.")
        if self.discord_enabled and not self.discord_webhook_url:
            print("[ALERT WARN] Discord enabled but no webhook URL configured.")
        if self.sms_enabled and not self.sms_recipients:
            print("[ALERT WARN] SMS enabled but no recipients configured.")

    def check_and_alert(self, event_type, name, confidence, details):
        """
        Check if an event warrants an alert and send notifications.

        Args:
            event_type: String event type (e.g., "DANGEROUS_OBJECT", "SPOOF_DETECTED")
            name: Source name (person name or object class)
            confidence: Float confidence score (0-1)
            details: String description of the event

        Returns:
            bool: True if alert was sent, False otherwise
        """
        if not self.enabled:
            return False

        # Determine severity
        severity, should_alert = self._classify_event(event_type, details)
        if not should_alert:
            return False

        # Check cooldown
        alert_key = f"{event_type}_{name}"
        cooldown = self.critical_cooldown_sec if severity == "CRITICAL" else self.alert_cooldown_sec
        if not self._can_alert(alert_key, cooldown):
            return False

        # Build notification content
        subject, body = self._build_message(event_type, name, confidence, details, severity)

        # Send to all enabled channels
        sent_count = 0

        if self.email_enabled:
            sent_count += int(self._send_email(subject, body, severity))

        if self.telegram_enabled:
            sent_count += int(self._send_telegram(subject, body, severity))

        if self.discord_enabled:
            sent_count += int(self._send_discord(subject, body, severity))

        if self.sms_enabled:
            sent_count += int(self._send_sms(subject, body, severity))

        if self.webhook_enabled:
            sent_count += int(self._send_webhook(event_type, name, confidence, details, severity))

        if sent_count > 0:
            self.last_alert_time[alert_key] = time.time()
            self.alert_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "name": name,
                "severity": severity,
                "channels_sent": sent_count,
            })
            # Keep history manageable
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]

            print(f"[ALERT] {severity}: {event_type} | {name} | {details}")
            print(f"[ALERT] Sent to {sent_count} channel(s)")

        return sent_count > 0

    def _classify_event(self, event_type, details):
        """
        Classify event severity and determine if alert should be sent.

        Returns:
            (severity_string, should_alert_bool)
            severity: "CRITICAL", "WARNING", or "INFO"
        """
        details_lower = (details or "").lower()

        # Check for dangerous objects in details
        is_dangerous_object = any(
            obj in details_lower for obj in self.DANGEROUS_OBJECTS
        )

        if event_type in self.CRITICAL_EVENTS or is_dangerous_object:
            return "CRITICAL", True

        if event_type in self.WARNING_EVENTS:
            return "WARNING", True

        return "INFO", False

    def _can_alert(self, key, cooldown):
        """Check if enough time has passed since last alert for this key."""
        last = self.last_alert_time.get(key, 0)
        return (time.time() - last) >= cooldown

    def _build_message(self, event_type, name, confidence, details, severity):
        """Build email/telegram subject and body."""
        subject = f"[{severity}] Security Alert: {event_type}"

        body = (
            f"{'='*50}\n"
            f"SECURITY ALERT - {severity}\n"
            f"{'='*50}\n\n"
            f"Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Event:      {event_type}\n"
            f"Source:     {name}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Details:    {details}\n"
            f"Severity:   {severity}\n\n"
            f"{'='*50}\n"
        )
        return subject, body

    # ------------------------------------------------------------------
    # Channel Senders
    # ------------------------------------------------------------------

    def _send_email(self, subject, body, severity):
        """Send email notification via SMTP."""
        if not self.email_to:
            return False
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            # FIX #1: self.port changed to self.smtp_port
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15)
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)
            server.send_message(msg)
            server.quit()
            return True
        except smtplib.SMTPAuthenticationError:
            print("[ALERT ERROR] Email auth failed. Check SMTP username/password.")
            self.email_enabled = False
            return False
        except Exception as e:
            print(f"[ALERT ERROR] Email failed: {e}")
            return False

    def _send_telegram(self, subject, body, severity):
        """Send Telegram message via Bot API."""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            text = f"*{subject}*\n\n```\n{body}\n```"

            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[ALERT ERROR] Telegram failed: {e}")
            return False

    def _send_discord(self, subject, body, severity):
        """Send Discord webhook embed."""
        try:
            colors = {"CRITICAL": 0xFF0000, "WARNING": 0xFFFF00, "INFO": 0x3498DB}
            color = colors.get(severity, 0x00FF00)

            # Discord embeds have a 4096 char limit for description
            desc = body[:4000] if len(body) > 4000 else body

            payload = {
                "embeds": [{
                    "title": f"Security Alert: {severity}",
                    "description": desc,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": "Security Detection System"},
                }]
            }
            resp = requests.post(self.discord_webhook_url, json=payload, timeout=10)
            return resp.status_code in (200, 204)
        except Exception as e:
            print(f"[ALERT ERROR] Discord failed: {e}")
            return False

    def _send_sms(self, subject, body, severity):
        """Send SMS via carrier email-to-SMS gateways.
        FIX #11: Reuse a single SMTP connection for all recipients instead of
        opening a new connection per recipient."""
        if not self.email_from or not self.smtp_user:
            print("[ALERT ERROR] SMS requires email SMTP to be configured.")
            return False
        try:
            sms_text = f"{subject}: {body[:100]}" if len(body) > 160 else subject
            sent = 0

            # Open ONE SMTP connection, then loop through recipients
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=15)
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)

            try:
                for recipient in self.sms_recipients:
                    number = recipient.get("number", "")
                    carrier = recipient.get("carrier", "").lower()
                    gateway = CARRIER_GATEWAYS.get(carrier, "")
                    if not gateway:
                        print(f"[ALERT WARN] Unknown carrier '{carrier}' for {number}. "
                              f"Supported: {', '.join(CARRIER_GATEWAYS.keys())}")
                        continue

                    to_addr = f"{number}{gateway}"
                    msg = MIMEText(sms_text)
                    msg["From"] = self.email_from
                    msg["To"] = to_addr
                    msg["Subject"] = subject

                    server.send_message(msg)
                    sent += 1
            finally:
                server.quit()

            return sent > 0
        except Exception as e:
            print(f"[ALERT ERROR] SMS failed: {e}")
            return False

    def _send_webhook(self, event_type, name, confidence, details, severity):
        """Send alert to a generic webhook endpoint."""
        try:
            payload = {
                "event_type": event_type,
                "source": name,
                "confidence": confidence,
                "details": details,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if self.webhook_method.upper() == "POST":
                resp = requests.post(
                    self.webhook_url, json=payload,
                    headers=self.webhook_headers, timeout=10
                )
            else:
                resp = requests.get(
                    self.webhook_url, params=payload,
                    headers=self.webhook_headers, timeout=10
                )
            return resp.status_code in (200, 201, 204)
        except Exception as e:
            print(f"[ALERT ERROR] Webhook failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_alert_history(self, limit=50):
        """Return recent alert history."""
        return self.alert_history[-limit:]

    def test_alert(self):
        """Send a test alert to all enabled channels."""
        print("[ALERT] Sending test alert...")
        subject = "[INFO] TEST - Security System Alert"
        body = (
            f"{'='*50}\n"
            f"THIS IS A TEST ALERT\n"
            f"{'='*50}\n\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"If you received this, your alert system is working!\n\n"
            f"{'='*50}\n"
        )

        # FIX #2: use underscore-prefixed private method names
        results = {}
        if self.email_enabled:
            results["Email"] = self._send_email("TEST Alert", body, "INFO")
        if self.telegram_enabled:
            results["Telegram"] = self._send_telegram("TEST Alert", body, "INFO")
        if self.discord_enabled:
            results["Discord"] = self._send_discord("TEST Alert", body, "INFO")
        if self.sms_enabled:
            results["SMS"] = self._send_sms("TEST Alert", body, "INFO")
        if self.webhook_enabled:
            results["Webhook"] = self._send_webhook("TEST", "System", 1.0, "Test alert", "INFO")

        for channel, success in results.items():
            status = "OK" if success else "FAILED"
            print(f"  {channel}: {status}")

        return all(results.values()) if results else False

    @staticmethod
    def create_sample_config(output_path="alert_config.json"):
        """Create a sample configuration file."""
        # FIX #3: key matches what __init__ reads: "critical_cooldown"
        sample = {
            "enabled": True,
            "alert_cooldown_sec": 300,
            "critical_cooldown": 60,
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "smtp_user": "matthew.arsene.en@gmail.com",
                "smtp_pass": "YOUR_SMTP_APP_PASSWORD",
                "from": "smartvision.alerts@gmail.com",
                "to": ["matthew.arsene.en@gmail.com"],
            },
            "telegram": {
                "enabled": False,
                "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
                "chat_id": "YOUR_CHAT_ID",
            },
            "discord": {
                "enabled": False,
                "webhook_url": "YOUR_DISCORD_WEBHOOK_URL",
            },
            "sms": {
                "enabled": False,
                "recipients": [
                    {"number": "1234567890", "carrier": "att"},
                ],
            },
            "webhook": {
                "enabled": False,
                "url": "https://your-webhook-url.com/endpoint",
                "method": "POST",
            },
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(sample, f, indent=4)

        print(f"[ALERT] Sample config written to: {output_path}")