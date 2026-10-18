#!/usr/bin/env python3
"""Tests for tokenizer-lean/chat_template.jinja against the stock template it was made from.

Run with Strata's venv (it has jinja2 and Strata's own ChatTemplate):
    ~/lab/Strata/.venv/bin/python ~/llama-swap/strata/test_lean_template.py
LEAN_TEMPLATE=<path> tests another copy.

The rule: reasoning_effort "low" carries the lean instruction in place of the stock low sentence; every other
request renders byte for byte as the stock template renders it.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, "/home/jim/lab/Strata/serve")
from server import ChatTemplate  # noqa: E402

HERE = Path(__file__).resolve().parent
STOCK = ChatTemplate(Path("/home/jim/lab/Strata-data/packs/iq3_s/tokenizer/chat_template.jinja"))
LEAN_PATH = Path(os.environ.get("LEAN_TEMPLATE", HERE / "tokenizer-lean" / "chat_template.jinja"))
LEAN = ChatTemplate(LEAN_PATH)
HAS_WARNINGS = "lean_ns" in LEAN_PATH.read_text()
STOCK_LOW = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
             "conclusion without unnecessary elaboration.")
LEAN_START = "Reasoning effort is set to lean."
WARN1 = "⚠️ SYSTEM WARNING: The previous tool call returned an error."
WARN2 = "consecutive tool errors detected."

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a command",
                                           "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]


def call(i, cmd):
    return {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": {"command": cmd}}}


def tool(i, text):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


CONVERSATIONS = {
    "single": [{"role": "user", "content": "Hi"}],
    "system": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
    "multi-turn": [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet back."},
        {"role": "user", "content": "Again"},
    ],
    "agent": [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the tests."},
        {"role": "assistant", "content": "", "reasoning_content": "Run them first.", "tool_calls": [call(1, "pytest")]},
        tool(1, "Traceback (most recent call last):\n  File \"x.py\", line 1\nNameError: name 'y' is not defined"),
        {"role": "assistant", "content": "", "reasoning_content": "Try again.", "tool_calls": [call(2, "pytest -x")]},
        tool(2, "Error: command failed\n\nCommand exited with code 1"),
        {"role": "assistant", "content": "", "reasoning_content": "Look at the file.", "tool_calls": [call(3, "cat x.py")]},
        tool(3, "print(y)\n"),
    ],
    "parallel-tools": [
        {"role": "user", "content": "Look around."},
        {"role": "assistant", "content": "", "tool_calls": [call(1, "ls"), call(2, "pwd")]},
        tool(1, "a.py\nb.py"),
        tool(2, "/tmp"),
    ],
}

UNCHANGED_LEVELS = [{}, {"reasoning_effort": "xhigh"}, {"reasoning_effort": "high"}, {"reasoning_effort": "medium"},
                    {"enable_thinking": False}, {"enable_thinking": True},
                    {"reasoning_effort": "low", "enable_thinking": False}]


def render(t, conv, tools, **kw):
    return t.render(CONVERSATIONS[conv], tools=tools, add_generation_prompt=True, **kw)


def lean_sentence(text):
    start = text.index(LEAN_START)
    return text[start:text.index(".", text.index("report what still fails")) + 1]


class OtherLevelsUnchanged(unittest.TestCase):
    def test_every_other_level_renders_identically(self):
        n = 0
        for conv in CONVERSATIONS:
            for tools in (None, TOOLS):
                for kw in UNCHANGED_LEVELS:
                    with self.subTest(conv=conv, tools=bool(tools), kw=kw):
                        self.assertEqual(render(LEAN, conv, tools, **kw), render(STOCK, conv, tools, **kw))
                        n += 1
        self.assertEqual(n, len(CONVERSATIONS) * 2 * len(UNCHANGED_LEVELS))


class Low(unittest.TestCase):
    def test_low_carries_the_lean_sentence_once(self):
        for conv in CONVERSATIONS:
            for tools in (None, TOOLS):
                out = render(LEAN, conv, tools, reasoning_effort="low")
                self.assertIn(LEAN_START, out)
                self.assertNotIn(STOCK_LOW, out)
                self.assertEqual(out.count("Reasoning effort is set to"), 1)

    def test_low_differs_from_stock_low_only_in_the_sentence(self):
        for conv in CONVERSATIONS:
            if HAS_WARNINGS and conv == "agent":
                continue                                  # the warnings are checked below
            for tools in (None, TOOLS):
                a = render(LEAN, conv, tools, reasoning_effort="low")
                b = render(STOCK, conv, tools, reasoning_effort="low")
                self.assertEqual(a.replace(lean_sentence(a), STOCK_LOW), b, conv)

    def test_low_carries_the_polishing_limit(self):
        out = render(LEAN, "agent", TOOLS, reasoning_effort="low")
        self.assertIn("at most two rounds and about ten minutes on reviewing and polishing", out)
        self.assertIn("does not apply to fixing real failures", out)
        for kw in UNCHANGED_LEVELS:
            self.assertNotIn("reviewing and polishing", render(LEAN, "agent", TOOLS, **kw), kw)

    def test_unknown_level_still_refused(self):
        with self.assertRaises(Exception):
            render(LEAN, "single", None, reasoning_effort="bogus")


@unittest.skipUnless(HAS_WARNINGS, "this template has no tool-error warnings")
class ToolErrorWarnings(unittest.TestCase):
    def results(self, *outputs):
        """The tool_response blocks of a low-effort render of one call per output, in order."""
        conv = [{"role": "user", "content": "Go."}]
        for i, out in enumerate(outputs):
            conv += [{"role": "assistant", "content": "", "tool_calls": [call(i, "x")]}, tool(i, out)]
        text = LEAN.render(conv, tools=TOOLS, add_generation_prompt=True, reasoning_effort="low")
        return [b.split("</tool_response>")[0] for b in text.split("<tool_response>")[1:]]

    def test_pi_exit_code_counts_and_escalates(self):
        r = self.results("boom\n\nCommand exited with code 1", "still\n\nCommand exited with code 2", "fine")
        self.assertIn(WARN1, r[0])
        self.assertIn("2 " + WARN2, r[1])
        self.assertNotIn("WARNING", r[2])

    def test_other_levels_never_warn(self):
        for kw in UNCHANGED_LEVELS:
            self.assertNotIn("SYSTEM WARNING", render(LEAN, "agent", TOOLS, **kw), kw)


if __name__ == "__main__":
    unittest.main(verbosity=1)
