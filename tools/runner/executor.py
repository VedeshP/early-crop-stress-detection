"""Execute a notebook headlessly while streaming its output live to a log file."""

import re
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _texts(msg):
    t, c = msg.get("msg_type"), msg.get("content", {})
    if t == "stream":
        return c.get("text", "")
    if t in ("execute_result", "display_data"):
        data = c.get("data", {})
        if "text/plain" in data and "image/png" not in data:
            return data["text/plain"] + "\n"
        if "image/png" in data:
            return "[figure]\n"
    if t == "error":
        return ANSI.sub("", "\n".join(c.get("traceback", []))) + "\n"
    return ""


def execute_notebook(src, dst, log_path, on_line=None, timeout=None, save_every=120):
    """Run notebook `src`, write the executed copy to `dst`. Returns (ok, error_text)."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, DeadKernelError

    src, dst = Path(src), Path(dst)
    nb = nbformat.read(src, as_version=4)
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    last_save = [time.time()]

    def emit(text):
        for line in text.splitlines():
            line = line.rstrip()
            log.write(line + "\n")
            if on_line:
                on_line(line)

    class LiveClient(NotebookClient):
        def output(self, outs, msg, display_id, cell_index):
            text = _texts(msg)
            if text:
                emit(text)
            if time.time() - last_save[0] > save_every:  # keep a partial copy for inspection
                last_save[0] = time.time()
                try:
                    nbformat.write(self.nb, dst)
                except Exception:
                    pass
            return super().output(outs, msg, display_id, cell_index)

    def on_start(cell, cell_index):
        if cell.cell_type == "code" and "toolkit-source" not in cell.metadata.get("tags", []):
            first = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
            emit(f">>> cell {cell_index}: {first[:100]}")

    client = LiveClient(nb, timeout=timeout, kernel_name="python3",
                        resources={"metadata": {"path": str(src.parent)}}, on_cell_start=on_start)
    ok, err = True, None
    try:
        client.execute()
    except CellExecutionError as e:
        ok, err = False, ANSI.sub("", str(e))[-3000:]
    except DeadKernelError as e:
        ok, err = False, f"kernel died (out of memory / crash?): {e}"
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    finally:
        try:
            nbformat.write(nb, dst)
        except Exception:
            pass
        log.close()
    return ok, err
