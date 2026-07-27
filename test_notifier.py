import smtplib
import json
import os
from datetime import datetime

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertManager:

    def __init__(self, config_path="alert_config.json"):
        self.config = {}

        # Load config
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

        email_cfg = self.config.get("email", {})

        self.enabled = self.config.get("enabled", True)

        self.email_enabled = email_cfg.get("enabled", False)

        self.smtp_server = email_cfg.get(
            "smtp_server",
            "smtp.gmail.com"
        )

        self.smtp_port = email_cfg.get(
            "smtp_port",
            587
        )

        self.smtp_user = email_cfg.get(
            "smtp_user",
            ""
        )

        self.smtp_pass = email_cfg.get(
            "smtp_pass",
            ""
        )

        self.email_from = email_cfg.get(
            "from",
            self.smtp_user
        )

        self.email_to = email_cfg.get(
            "to",
            []
        )

        if isinstance(self.email_to, str):
            self.email_to = [self.email_to]

        print("[INFO] AlertManager initialized")

    def _send_email(self, subject, body):

        try:
            print("[INFO] Creating email message...")

            msg = MIMEMultipart()

            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            print("[INFO] Connecting to SMTP server...")

            server = smtplib.SMTP(
                self.smtp_server,
                self.smtp_port,
                timeout=20
            )

            print("[INFO] Starting TLS encryption...")
            server.starttls()

            print("[INFO] Logging into Gmail...")
            server.login(self.smtp_user, self.smtp_pass)

            print("[INFO] Sending email...")
            server.send_message(msg)

            print("[INFO] Closing SMTP connection...")
            server.quit()

            print("[SUCCESS] Email sent")
            return True

        except smtplib.SMTPAuthenticationError as e:
            print("\n[AUTH ERROR]")
            print("Gmail rejected login.")
            print("Possible causes:")
            print("1. Wrong app password")
            print("2. 2-Factor Authentication not enabled")
            print("3. Using normal Gmail password instead of App Password")
            print(f"\nError: {e}")
            return False

        except smtplib.SMTPConnectError as e:
            print("\n[SMTP CONNECTION ERROR]")
            print(e)
            return False

        except smtplib.SMTPException as e:
            print("\n[SMTP ERROR]")
            print(e)
            return False

        except Exception as e:
            print("\n[GENERAL ERROR]")
            print(type(e).__name__)
            print(e)
            return False

    def test_alert(self):

        subject = "[TEST] Smart Vision Alert"

        body = f"""
SMART VISION SECURITY SYSTEM

This is a test email.

Time:
{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

If you received this email,
the notification system is working correctly.
"""

        return self._send_email(subject, body)

    @staticmethod
    def create_sample_config(output_path="alert_config.json"):

        # ONLY CREATE IF FILE DOES NOT EXIST
        if os.path.exists(output_path):
            return

        sample = {
            "enabled": True,

            "email": {
                "enabled": True,

                "smtp_server": "smtp.gmail.com",

                "smtp_port": 587,

                # YOUR GMAIL
                "smtp_user": "matthew.arsene.en@gmail.com",

                # GMAIL APP PASSWORD
                "smtp_pass": "YOUR_SMTP_APP_PASSWORD",

                # SENDER
                "from": "matthew.arsene.en@gmail.com",

                # RECEIVER
                "to": [
                    "matthew.arsene.en@gmail.com"
                ]
            }
        }

        with open(output_path, "w") as f:
            json.dump(sample, f, indent=4)

        print(f"[INFO] Created sample config: {output_path}")