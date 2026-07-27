from test_notifier import AlertManager

def main():
    print("=" * 60)
    print(" SMART VISION EMAIL ALERT TEST ")
    print("=" * 60)

    CONFIG_PATH = "alert_config.json"

    # Create config automatically if it doesn't exist
    AlertManager.create_sample_config(CONFIG_PATH)

    # Load alert manager
    alert_mgr = AlertManager(config_path=CONFIG_PATH)

    print("\n[INFO] Loaded configuration")
    print(f"[INFO] Email Enabled: {alert_mgr.email_enabled}")
    print(f"[INFO] SMTP Server: {alert_mgr.smtp_server}")
    print(f"[INFO] Sender: {alert_mgr.email_from}")
    print(f"[INFO] Receiver(s): {alert_mgr.email_to}")

    print("\n[TEST] Sending test email...\n")

    success = alert_mgr.test_alert()

    if success:
        print("\n[SUCCESS] Email sent successfully.")
        print("Check:")
        print("1. Inbox")
        print("2. Spam/Junk folder")
        print("3. Promotions tab (for Gmail)")
    else:
        print("\n[FAILURE] Email was not sent.")
        print("Read the error logs above carefully.")


if __name__ == "__main__":
    main()