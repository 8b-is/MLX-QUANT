# Explicit email helper

`send_email.py` has no import-time side effects. Sending requires `--send` and
an explicit server, sender, recipient, subject and UTF-8 body file. There is no
saved recipient or benchmark report.

Provide `MLX_SMTP_USERNAME` and `MLX_SMTP_PASSWORD` through your private local
environment or secret manager. Do not put credentials in the script, command
arguments, repository or report body. The helper requires STARTTLS with the
system's certificate verification; it does not fall back to plaintext or ignore
certificate errors.

After reviewing the intended recipient and report, invoke it explicitly:

```sh
python3 mlx-quant-linux/send_email.py --send \
  --smtp-host smtp.example.test --smtp-port 587 \
  --from sender@example.test --to recipient@example.test \
  --subject 'Reviewed project report' --body-file report.txt
```

The addresses above are placeholders. `--help` opens no connection. Missing
arguments or credentials fail before connecting; transport failures return a
nonzero exit status without printing server responses or credentials.

Run the offline tests without MLX, models or network access:

```sh
python3 -m unittest discover -s scripts -p test_send_email.py -v
```

Removing a credential from current source does not revoke it or remove it from
Git history. Its owner must separately confirm revocation or rotation if it was
live. Do not test the historical credential to determine whether it still works.
