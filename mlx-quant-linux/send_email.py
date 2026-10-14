"""Explicit email helper. Importing this module never sends a message."""

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true", required=True)
    parser.add_argument("--smtp-host", required=True)
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--from", dest="sender", required=True)
    parser.add_argument("--to", dest="recipient", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.smtp_port <= 65535:
        parser.error("--smtp-port must be between 1 and 65535")

    username = os.environ.get("MLX_SMTP_USERNAME")
    password = os.environ.get("MLX_SMTP_PASSWORD")
    if not username or not password:
        parser.error("Set MLX_SMTP_USERNAME and MLX_SMTP_PASSWORD before sending")

    try:
        message = EmailMessage()
        message.set_content(args.body_file.read_text(encoding="utf-8"))
        message["From"] = args.sender
        message["To"] = args.recipient
        message["Subject"] = args.subject
        context = ssl.create_default_context()
        with smtplib.SMTP(args.smtp_host, args.smtp_port, timeout=10) as client:
            client.starttls(context=context)
            client.login(username, password)
            client.send_message(message)
    except (OSError, ValueError, smtplib.SMTPException) as error:
        # SMTP error responses can echo credentials or private message data.
        print(f"Email not sent ({type(error).__name__}).", file=sys.stderr)
        return 1
    print("Sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
