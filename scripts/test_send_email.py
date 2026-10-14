"""Email helper tests with synthetic data and a mocked SMTP transport only."""

import contextlib
import importlib.util
import io
import os
import smtplib
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "mlx-quant-linux/send_email.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("email_helper_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmailHelperTests(unittest.TestCase):
    def setUp(self):
        # Guard module import too: no original or fixed helper may open a socket.
        smtp_patch = patch.object(smtplib, "SMTP")
        self.smtp = smtp_patch.start()
        self.addCleanup(smtp_patch.stop)
        self.helper = load_helper()
        self.smtp.assert_not_called()
        environment = patch.dict(
            os.environ,
            {
                "MLX_SMTP_USERNAME": "synthetic-user",
                "MLX_SMTP_PASSWORD": "test-only-value",
            },
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.body = Path(directory.name) / "report.txt"
        self.body.write_text("Synthetic report.\n", encoding="utf-8")
        self.args = [
            "--send",
            "--smtp-host",
            "smtp.example.test",
            "--from",
            "sender@example.test",
            "--to",
            "recipient@example.test",
            "--subject",
            "Synthetic subject",
            "--body-file",
            str(self.body),
        ]
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.client = self.smtp.return_value.__enter__.return_value

    def invoke(self, args=None):
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(
            self.stderr
        ):
            return self.helper.main(self.args if args is None else args)

    def test_import_has_no_transport_side_effect(self):
        load_helper()
        self.smtp.assert_not_called()

    def test_help_and_missing_send_never_connect(self):
        for args, code in [(["--help"], 0), (self.args[1:], 2), ([], 2)]:
            with self.subTest(args=args):
                with self.assertRaises(SystemExit) as caught:
                    self.invoke(args)
                self.assertEqual(caught.exception.code, code)
                self.smtp.assert_not_called()

    def test_missing_credentials_never_connect(self):
        for key in ["MLX_SMTP_USERNAME", "MLX_SMTP_PASSWORD"]:
            with self.subTest(key=key), patch.dict(os.environ, {key: ""}):
                with self.assertRaises(SystemExit) as caught:
                    self.invoke()
                self.assertEqual(caught.exception.code, 2)
                self.smtp.assert_not_called()

    def test_invalid_port_never_connects(self):
        for port in ["0", "65536", "not-a-port"]:
            with self.subTest(port=port), self.assertRaises(SystemExit):
                self.invoke(self.args + ["--smtp-port", port])
            self.smtp.assert_not_called()

    def test_missing_body_never_connects(self):
        self.body.unlink()
        self.assertEqual(self.invoke(), 1)
        self.smtp.assert_not_called()

    def test_header_injection_never_connects(self):
        args = list(self.args)
        args[args.index("--subject") + 1] = "Subject\nBcc: someone@example.test"
        self.assertEqual(self.invoke(args), 1)
        self.smtp.assert_not_called()

    def test_explicit_send_uses_validated_tls_and_supplied_message(self):
        self.assertEqual(self.invoke(), 0)
        self.smtp.assert_called_once_with("smtp.example.test", 587, timeout=10)
        context = self.client.starttls.call_args.kwargs["context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.client.login.assert_called_once_with("synthetic-user", "test-only-value")
        message = self.client.send_message.call_args.args[0]
        self.assertEqual(message["From"], "sender@example.test")
        self.assertEqual(message["To"], "recipient@example.test")
        self.assertEqual(message["Subject"], "Synthetic subject")
        self.assertEqual(message.get_content(), "Synthetic report.\n")
        self.assertEqual(
            [c[0] for c in self.client.method_calls],
            ["starttls", "login", "send_message"],
        )
        self.smtp.return_value.__exit__.assert_called_once()
        self.assertEqual(self.stdout.getvalue(), "Sent\n")

    def test_tls_failure_does_not_authenticate_or_send(self):
        self.client.starttls.side_effect = ssl.SSLCertVerificationError(
            "synthetic TLS failure"
        )
        self.assertEqual(self.invoke(), 1)
        self.client.login.assert_not_called()
        self.client.send_message.assert_not_called()
        self.smtp.return_value.__exit__.assert_called_once()

    def test_auth_failure_does_not_send_or_echo_server_response(self):
        self.client.login.side_effect = smtplib.SMTPAuthenticationError(
            535, b"test-only-value"
        )
        self.assertEqual(self.invoke(), 1)
        self.client.send_message.assert_not_called()
        self.assertNotIn("test-only-value", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")
        self.smtp.return_value.__exit__.assert_called_once()

    def test_send_failure_closes_transport_and_returns_failure(self):
        self.client.send_message.side_effect = smtplib.SMTPException(
            "private synthetic body"
        )
        self.assertEqual(self.invoke(), 1)
        self.assertNotIn("private synthetic body", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")
        self.smtp.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
