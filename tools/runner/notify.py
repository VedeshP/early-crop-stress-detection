"""Notifications to your phone / chat / email. Standard library only; failures never stop a run.

Channels (enable any number in configs/notify.json):
  ntfy      push notifications via the free ntfy app (https://ntfy.sh) - no account needed
  telegram  a Telegram bot you create with @BotFather
  discord   a Discord channel webhook
  slack     a Slack incoming webhook
  email     any SMTP server (e.g. Gmail with an app password)
  file      append to a text file;  console: print
"""

import json
import smtplib
import ssl
import time
import urllib.request
from email.message import EmailMessage
from pathlib import Path

PRIORITY = {"low": 2, "default": 3, "high": 4, "urgent": 5}


def _post(url, data, headers=None, timeout=20):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _post_json(url, payload, timeout=20):
    return _post(url, json.dumps(payload).encode(), {"Content-Type": "application/json"}, timeout)


class Notifier:
    def __init__(self, config=None, run_label="agrivision"):
        self.cfg = config or {}
        self.channels = {k: v for k, v in (self.cfg.get("channels") or {}).items() if v and v.get("enabled", True)}
        self.label = run_label
        self.min_gap = float(self.cfg.get("min_seconds_between_same", 300))
        self._last = {}
        self.errors = []

    @classmethod
    def from_file(cls, path, run_label="agrivision"):
        if not path:
            return cls({"channels": {"console": {}}}, run_label)
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), run_label)

    def send(self, title, message, priority="default", tags=(), key=None):
        """Send to every enabled channel. `key` rate-limits repeats of the same alert."""
        if key:
            now = time.time()
            if now - self._last.get(key, 0) < self.min_gap:
                return
            self._last[key] = now
        title = f"[{self.label}] {title}"
        for name, ch in self.channels.items():
            try:
                getattr(self, f"_{name}")(ch, title, message, priority, tags)
            except Exception as e:  # never let a notification failure break the run
                err = f"{time.strftime('%H:%M:%S')} {name}: {type(e).__name__}: {e}"
                self.errors.append(err)
                print(f"[notify] failed: {err}", flush=True)

    # ---- channels
    def _ntfy(self, ch, title, message, priority, tags):
        server = ch.get("server", "https://ntfy.sh").rstrip("/")
        headers = {"Title": title.encode("utf-8").decode("latin-1", "ignore"),
                   "Priority": str(PRIORITY.get(priority, 3))}
        if tags:
            headers["Tags"] = ",".join(tags)
        if ch.get("click"):
            headers["Click"] = ch["click"]
        if ch.get("token"):
            headers["Authorization"] = f"Bearer {ch['token']}"
        _post(f"{server}/{ch['topic']}", message.encode("utf-8"), headers)

    def _telegram(self, ch, title, message, priority, tags):
        _post_json(f"https://api.telegram.org/bot{ch['bot_token']}/sendMessage",
                   {"chat_id": ch["chat_id"], "text": f"{title}\n{message}",
                    "disable_notification": priority == "low"})

    def _discord(self, ch, title, message, priority, tags):
        _post_json(ch["webhook_url"], {"content": f"**{title}**\n{message}"[:1990]})

    def _slack(self, ch, title, message, priority, tags):
        _post_json(ch["webhook_url"], {"text": f"*{title}*\n{message}"})

    def _email(self, ch, title, message, priority, tags):
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = title, ch["from"], ch["to"]
        msg.set_content(message)
        host, port = ch.get("host", "smtp.gmail.com"), int(ch.get("port", 465))
        if ch.get("starttls"):
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(ch["username"], ch["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as s:
                s.login(ch["username"], ch["password"])
                s.send_message(msg)

    def _file(self, ch, title, message, priority, tags):
        with open(ch.get("path", "notifications.log"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{priority}] {title}\n{message}\n\n")

    def _console(self, ch, title, message, priority, tags):
        print(f"\n*** {title}\n{message}\n", flush=True)
