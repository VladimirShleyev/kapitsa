#!/usr/bin/env python3
"""The request set for the KFlash candidate decoder's training data (internal/KFLASH.md §3, M2): a JSONL
of requests for `kflash_datagen gen`, built from local sources only. Stdlib only.

Domains (meta.domain), each split into train and held-out (meta.split) by source, never by answer:
  agent      the recorded opencode session (bench/harness/agent-traces/opencode-ledger.json), its requests as the
             template renders them with tools, thinking off; held out: requests 6, 12 and 18
  agent_syn  coding-agent turns over local source files: the session's system prompt and tools, a task, the agent's
             read of the file and the file as the read tool returns it; the model writes the next turn (an edit, a
             test file, an explanation). Held out by file
  code       code to write (tasks x languages) and code to read (explain / review / test / port a local snippet)
  chat       explanations, writing, advice, summaries of package READMEs, translations
  math       word problems with random numbers, thinking on (reasoning traces)
  bench      held out only: the speed benchmark's prompts (bench/prompts code_27b, p700, p4000)
Train prompts get a greedy answer or one sampled at T = 0.6 (agent: greedy and three seeds); held-out prompts a
greedy answer only (the offline evaluation's greedy acceptance is exact on the target's greedy path).

usage: prompts.py --out requests.jsonl [--scale 1.0] [--seed 7]
"""
import argparse, glob, hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TRACE = os.path.join(REPO, "bench/harness/agent-traces/opencode-ledger.json")
SITE = os.path.expanduser("~/Desktop/Nace Agent/research/inference/venvs/mlxlm/lib/python3.12/site-packages")
HELD_AGENT = {6, 12, 18}


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def held_by(key, frac):
    return (h(key) % 1000) < frac * 1000


# ---------------------------------------------------------------- sources
def code_files():
    """(display path, text, language) of local source files: kapitsa, mlx_lm, transformers and other packages."""
    pats = [(REPO, "src/**/*.cpp", "C++"), (REPO, "src/**/*.h", "C++"), (REPO, "src/kernels/**/*.metal", "Metal"),
            (REPO, "tools/*.py", "Python"), (REPO, "bench/**/*.py", "Python"), (REPO, "tools/*.cpp", "C++")]
    for pkg in ["mlx_lm", "transformers", "huggingface_hub", "pydantic", "rich", "jinja2", "click", "httpx2",
                "openai", "pygments", "anyio", "tqdm", "packaging", "markdown_it", "filelock", "fsspec"]:
        pats.append((SITE, pkg + "/**/*.py", "Python"))
    out = []
    for root, pat, lang in pats:
        for p in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
            if "/tests/" in p or "__pycache__" in p:
                continue
            try:
                txt = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if 1500 < len(txt) < 400000:
                out.append((os.path.relpath(p, root), txt, lang))
    return out


def excerpt(text, rng, lo=60, hi=260):
    lines = text.split("\n")
    if len(lines) <= hi:
        return 1, lines
    n = rng.randint(lo, hi)
    a = rng.randint(0, len(lines) - n)
    return a + 1, lines[a:a + n]


def functions(lines, lang):
    rx = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(") if lang == "Python" else \
        re.compile(r"^[A-Za-z_][\w:<>,\s\*&]*?\b([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?(?:override\s*)?\{?\s*$")
    names = []
    for ln in lines:
        m = rx.match(ln)
        if m and m.group(1) not in ("if", "for", "while", "switch", "return", "main", "__init__"):
            names.append(m.group(1))
    return names


AGENT_TASKS = [
    "Add a docstring to every function in {path} that does not have one.",
    "Rename `{fn}` in {path} to `{fn2}` and update every use of it in that file.",
    "Add type hints to the functions in {path}.",
    "Refactor `{fn}` in {path} so it is easier to read, without changing what it does.",
    "Write unit tests for `{fn}` from {path} in a new test file.",
    "Explain what {path} does and how its main pieces fit together.",
    "Make `{fn}` in {path} validate its arguments and raise a clear error on bad input.",
    "Add debug logging to `{fn}` in {path}.",
    "`{fn}` in {path} is slow on large inputs. Find out why and make it faster.",
    "There is a bug in `{fn}` in {path}: it misbehaves on empty input. Find it and fix it.",
    "Add comments to the tricky parts of `{fn}` in {path}.",
    "Split `{fn}` in {path} into smaller helper functions.",
    "Review {path} and fix any bugs you find.",
    "Remove the dead code from {path}.",
    "Add a command-line flag to turn off the cache in {path}.",
    "Update `{fn}` in {path} to also accept a list of inputs.",
]
CPP_TASKS = [
    "Add a short comment above every function in {path} that has none.",
    "Rename `{fn}` in {path} to `{fn2}` and update every use of it in that file.",
    "Refactor `{fn}` in {path} so it is easier to read, without changing what it does.",
    "Explain what {path} does and how its main pieces fit together.",
    "Make `{fn}` in {path} check its arguments and fail with a clear message on bad input.",
    "Add a unit test for `{fn}` from {path}.",
    "Review {path} for memory-safety problems and fix the ones you find.",
    "`{fn}` in {path} allocates on every call. Make it reuse its buffers.",
    "Add timing instrumentation around `{fn}` in {path}.",
    "Review {path} and fix any bugs you find.",
]
LEADS = [None, "Let me read the file first.", "I'll start by reading the file.", "Let me look at the code.",
         "I'll take a look at the current implementation."]


def agent_syn(trace, files, rng, n_train, n_held):
    sys_prompt = trace["messages"][0]["content"].replace("qwen3.6-35b-a3b", "qwen3.5-4b")
    tools = trace["tool_sets"][0]
    out = []
    want = {"train": n_train, "heldout": n_held}
    got = {"train": 0, "heldout": 0}
    order = list(range(len(files)))
    rng.shuffle(order)
    for fi in order:
        path, text, lang = files[fi]
        split = "heldout" if held_by("file:" + path, 0.1) else "train"
        if got[split] >= want[split]:
            if all(got[k] >= want[k] for k in want):
                break
            continue
        start, lines = excerpt(text, rng)
        fns = functions(lines, lang) or ["main"]
        fn = rng.choice(fns)
        fn2 = fn + rng.choice(["_impl", "_v2", "_checked", "_fast"]) if not fn.startswith("_") else fn.lstrip("_")
        apath = "/work/repo/" + path
        task = rng.choice(AGENT_TASKS if lang == "Python" else CPP_TASKS).format(path=apath, fn=fn, fn2=fn2)
        cid = "call_" + hashlib.sha1((path + task).encode()).hexdigest()[:24]
        body = f"Read file {apath}, lines {start}-{start + len(lines) - 1}\n" + "\n".join(
            f"{start + i}: {ln}" for i, ln in enumerate(lines))
        args = {"path": apath} if start == 1 else {"path": apath, "offset": start, "limit": len(lines)}
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": task},
                {"role": "assistant", "content": rng.choice(LEADS),
                 "tool_calls": [{"id": cid, "type": "function",
                                 "function": {"name": "read", "arguments": json.dumps(args, separators=(",", ":"))}}]},
                {"role": "tool", "tool_call_id": cid, "content": body}]
        got[split] += 1
        out.append({"id": f"agent_syn-{len(out)}", "messages": msgs, "tools": tools,
                    "template_kwargs": {"enable_thinking": False}, "max_tokens": 768,
                    "meta": {"domain": "agent_syn", "split": split, "src": path}})
    return out


# Unseen agent turns (--unseen): task phrasings and a tool flow (grep, then read) that the training set never uses,
# on held-out files only, to measure what the decoder learned beyond the training templates
UNSEEN_TASKS = [
    "Which functions in {path} are never called from inside that file?",
    "I keep getting an exception from `{fn}` in {path} on some inputs. What is wrong and how do I fix it?",
    "Rewrite `{fn}` in {path} with early returns instead of nested conditionals.",
    "Make `{fn}` in {path} safe to call from several threads at once.",
    "Write a short section for the docs that explains what {path} provides.",
    "Replace the magic numbers in `{fn}` ({path}) with named constants.",
    "Could `{fn}` in {path} ever loop forever or recurse without end? Check, and fix it if so.",
    "Pull the error handling out of `{fn}` in {path} into a helper and reuse it.",
    "Port `{fn}` from {path} to TypeScript in a new file next to it.",
    "Add a verbose mode that reports what `{fn}` in {path} does at each step.",
    "`{fn}` in {path} is hard to test. Restructure it so its logic can be tested without I/O.",
    "Find every place in {path} that could raise on a None value and guard it.",
]


def agent_unseen(trace, files, rng, n):
    sys_prompt = trace["messages"][0]["content"].replace("qwen3.6-35b-a3b", "qwen3.5-4b")
    tools = trace["tool_sets"][0]
    out = []
    order = [i for i in range(len(files)) if held_by("file:" + files[i][0], 0.1)]
    rng.shuffle(order)
    for fi in order[:n]:
        path, text, lang = files[fi]
        start, lines = excerpt(text, rng, 60, 220)
        fn = rng.choice(functions(lines, lang) or ["main"])
        apath = "/work/repo/" + path
        task = rng.choice(UNSEEN_TASKS).format(path=apath, fn=fn)
        g_id = "call_" + hashlib.sha1(("g" + path + task).encode()).hexdigest()[:24]
        r_id = "call_" + hashlib.sha1(("r" + path + task).encode()).hexdigest()[:24]
        hits = [(start + i, ln) for i, ln in enumerate(lines) if fn in ln][:6] or [(start, lines[0])]
        grep_out = f"Found {len(hits)} matches\n{apath}:\n" + "\n".join(f"  Line {k}: {ln.strip()}" for k, ln in hits)
        body = f"Read file {apath}, lines {start}-{start + len(lines) - 1}\n" + "\n".join(
            f"{start + i}: {ln}" for i, ln in enumerate(lines))
        args = {"path": apath} if start == 1 else {"path": apath, "offset": start, "limit": len(lines)}
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": task},
                {"role": "assistant", "content": None, "tool_calls": [{"id": g_id, "type": "function", "function": {
                    "name": "grep", "arguments": json.dumps({"pattern": fn, "path": "/work/repo"}, separators=(",", ":"))}}]},
                {"role": "tool", "tool_call_id": g_id, "content": grep_out},
                {"role": "assistant", "content": None, "tool_calls": [{"id": r_id, "type": "function", "function": {
                    "name": "read", "arguments": json.dumps(args, separators=(",", ":"))}}]},
                {"role": "tool", "tool_call_id": r_id, "content": body}]
        out.append({"id": f"agent_unseen-{len(out)}", "messages": msgs, "tools": tools,
                    "template_kwargs": {"enable_thinking": False}, "max_tokens": 768, "temperature": 0.0, "seed": 0,
                    "meta": {"domain": "agent_unseen", "split": "heldout", "src": path}})
    return out


WRITE_TASKS = [
    "an LRU cache with O(1) get and put", "a function that parses ISO-8601 durations like P3DT4H", "a thread-safe bounded queue",
    "a command-line tool that counts word frequencies in a set of files", "binary search over a rotated sorted array",
    "a trie with insert, search and prefix search", "Dijkstra's shortest-path algorithm on an adjacency list",
    "a JSON pretty-printer that does not use a JSON library", "a token-bucket rate limiter",
    "a function that merges overlapping intervals", "a cache-friendly tiled matrix multiplication",
    "a minimal HTTP server that serves files from a directory", "a debounce helper",
    "a converter from a small Markdown subset (headings, lists, emphasis) to HTML", "a stack class with unit tests",
    "an e-mail address validator", "a CSV parser that handles quoted fields and escaped quotes",
    "topological sort with cycle detection", "a tokenizer and evaluator for arithmetic expressions with parentheses",
    "a binary min-heap priority queue", "a Bloom filter", "run-length encoding and decoding",
    "a Sudoku board validator", "the Levenshtein edit distance", "the skeleton of a URL-shortener web service",
    "a producer/consumer pipeline with worker threads", "a retry decorator with exponential backoff and jitter",
    "a function that flattens arbitrarily nested lists", "a 2-D vector type with arithmetic operators",
    "serialization and deserialization of a binary tree", "a ring buffer", "a union-find structure with path compression",
    "an interval tree", "a simple key-value store persisted to a file", "a function that formats byte counts (KiB, MiB)",
    "a parser for command-line flags", "a matrix determinant by Gaussian elimination", "quicksort with a median-of-three pivot",
    "a k-means clustering routine", "a function that checks balanced brackets", "a simple regular-expression matcher supporting . and *",
    "a conversion between Roman numerals and integers", "a calendar-month printer", "an in-memory event bus with subscribe/publish",
    "a function computing the n-th Fibonacci number modulo 1e9+7 in O(log n)", "a sliding-window maximum",
    "a generic memoization wrapper", "a breadth-first search on a grid with obstacles", "a simple bank account class with transfers",
    "a function that pretty-prints a table of rows", "a SHA-256 file checksum utility", "a word-wrap function",
    "a text-based tic-tac-toe game", "a scheduler that runs tasks at fixed intervals", "an A* path finder on a grid",
    "a function that detects a cycle in a linked list", "a reader for INI configuration files", "a big-integer addition",
    "the game of life on a toroidal grid", "a histogram of line lengths in a file", "a function to diff two lists of lines",
]
LANGS = ["Python", "Python", "Python", "C++", "C++", "Rust", "Go", "TypeScript", "JavaScript", "Java", "C", "Swift"]
WRITE_FORMS = ["Write {t} in {l}.", "Implement {t} in {l}. Include a short usage example.",
               "In {l}, write {t}. Add comments where it helps.", "Can you write {t} in {l}?",
               "I need {t} in {l}. Keep it simple and idiomatic.", "Show me how to write {t} in {l}, with tests."]
READ_TASKS = ["Explain what this code does:", "Review this code and point out bugs or risky parts:",
              "Write unit tests for this code:", "Add docstrings or comments to this code:",
              "Simplify this code without changing its behavior:", "Port this code to {l2}:",
              "What are the time and memory costs of this code? Suggest improvements:",
              "Rewrite this code to be more idiomatic:", "Summarize this file for a new contributor:"]
SQL_TASKS = ["the top 3 salaries per department", "customers who ordered every product in a category",
             "a running total of daily sales", "duplicate e-mail addresses in a users table",
             "the median order value per month", "users who were active on three consecutive days"]
BASH_TASKS = ["backs up a directory with rotation of the last 7 copies", "finds the 10 largest files under a path",
              "watches a log file and alerts on the word ERROR", "renames photos by their date",
              "checks that a list of URLs responds with HTTP 200"]


def code_requests(files, rng, n_write, n_read, held_frac=0.1):
    out = []
    for i in range(n_write):
        r = rng.random()
        if r < 0.08:
            t = rng.choice(SQL_TASKS)
            text = f"Write an SQL query that finds {t}. Explain how it works."
            key = "sql:" + t
        elif r < 0.14:
            t = rng.choice(BASH_TASKS)
            text = f"Write a bash script that {t}."
            key = "bash:" + t
        else:
            t, l = rng.choice(WRITE_TASKS), rng.choice(LANGS)
            text = rng.choice(WRITE_FORMS).format(t=t, l=l)
            key = "write:" + t
        split = "heldout" if held_by(key, held_frac) else "train"
        out.append({"id": f"code_write-{i}", "messages": [{"role": "user", "content": text}],
                    "template_kwargs": {"enable_thinking": False}, "max_tokens": 768,
                    "meta": {"domain": "code", "split": split, "src": key}})
    for i in range(n_read):
        path, text, lang = rng.choice(files)
        start, lines = excerpt(text, rng, 25, 110)
        task = rng.choice(READ_TASKS).format(l2=rng.choice([x for x in LANGS if x != lang]))
        fence = {"Python": "python", "C++": "cpp", "Metal": "cpp"}.get(lang, "")
        body = f"{task}\n\n```{fence}\n" + "\n".join(lines) + "\n```"
        split = "heldout" if held_by("file:" + path, held_frac) else "train"
        out.append({"id": f"code_read-{i}", "messages": [{"role": "user", "content": body}],
                    "template_kwargs": {"enable_thinking": False}, "max_tokens": 640,
                    "meta": {"domain": "code", "split": split, "src": path}})
    return out


TOPICS = ["how vaccines train the immune system", "why the sky is blue", "how a CPU cache works", "the causes of the French Revolution",
          "how compound interest works", "what a hash table is", "how plate tectonics shapes continents", "the idea behind public-key cryptography",
          "how photosynthesis stores energy", "what inflation is and why central banks care", "how a transformer language model generates text",
          "the difference between TCP and UDP", "how GPS finds your position", "why we have seasons", "how a bill becomes law in the United States",
          "what entropy means in thermodynamics", "how garbage collection works in managed languages", "the history of the printing press",
          "how noise-cancelling headphones work", "the basics of supply and demand", "how the heart pumps blood", "what a black hole is",
          "how a database index speeds up queries", "the Silk Road's role in trade", "how batteries store energy", "why bread rises",
          "the rules of chess for a beginner", "how a compiler turns code into machine instructions", "what climate feedback loops are",
          "how airplanes stay in the air", "the difference between a virus and a bacterium", "how recommendation systems work",
          "the main ideas of stoicism", "how earthquakes are measured", "how version control with git works", "the water cycle"]
CHAT_FORMS = ["Explain {t} in simple terms.", "Give me a clear, short explanation of {t}.",
              "I'm a high-school student. Can you explain {t}?", "Write a short blog post about {t}.",
              "What are the most common misconceptions about {t}?", "Explain {t} with an analogy, then in technical terms."]
WRITING = ["Write a polite e-mail asking my landlord to fix a leaking tap.", "Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Write a haiku sequence about autumn in the city.", "Draft a product description for a lightweight hiking backpack.",
           "Write a cover letter for a junior data analyst position.", "Write a toast for my sister's wedding.",
           "Write a limerick about a cat who learns to code.", "Draft an agenda for a one-hour project kickoff meeting.",
           "Write a short dialogue between a detective and a suspect.", "Write a motivational note for a friend starting a new job.",
           "Write release notes for version 2.0 of a note-taking app.", "Write a short speech about the importance of public libraries.",
           "Draft a complaint to an airline about a lost suitcase.", "Write a bedtime story about a dragon who is afraid of the dark.",
           "Write a LinkedIn post announcing that our team shipped a new feature.", "Write a poem about the ocean in the style of a sonnet."]
ADVICE = ["How should I prepare for a technical interview in two weeks?", "What is a good routine to learn a new language as an adult?",
          "How can I make my small apartment feel more spacious?", "How do I start running if I have never exercised?",
          "What should I consider before adopting a dog?", "How can I give constructive feedback to a coworker?",
          "How do I plan a budget for a two-week trip to Japan?", "What are good habits for working from home?",
          "How can I get better sleep?", "How do I negotiate a higher salary?", "How should I organize my notes for a research project?",
          "What are the pros and cons of renting versus buying a home?"]
TRANSLATE = ["French", "Spanish", "German", "Chinese", "Russian", "Japanese", "Italian", "Portuguese"]
PARAS = ["The meeting has been moved to Thursday afternoon because several team members are travelling on Wednesday. Please update your calendars and let me know if the new time does not work for you.",
         "Our new library opens on Monday. It has a children's corner, a quiet study floor and a café, and it will stay open until nine in the evening on weekdays.",
         "To reset your password, open the settings page, choose 'Security', and follow the link we send to your e-mail address. The link expires after one hour.",
         "The weather forecast predicts heavy rain over the weekend, so the outdoor concert will take place in the town hall instead."]


def readmes(rng):
    out = []
    for meta in sorted(glob.glob(os.path.join(SITE, "*.dist-info/METADATA"))):
        txt = open(meta, encoding="utf-8", errors="replace").read()
        body = txt.split("\n\n", 1)[1] if "\n\n" in txt else ""
        if len(body) > 1500:
            a = rng.randint(0, max(0, len(body) - 6000))
            out.append((os.path.basename(os.path.dirname(meta)), body[a:a + rng.randint(2500, 6000)]))
    return out


def chat_requests(rng, n, held_frac=0.2):
    docs = readmes(rng)
    out = []
    for i in range(n):
        r = rng.random()
        if r < 0.35:
            t = rng.choice(TOPICS)
            text, key = rng.choice(CHAT_FORMS).format(t=t), "topic:" + t
        elif r < 0.55:
            text = rng.choice(WRITING)
            key = "write:" + text
        elif r < 0.7:
            text = rng.choice(ADVICE)
            key = "advice:" + text
        elif r < 0.82 and docs:
            name, doc = rng.choice(docs)
            text = rng.choice(["Summarize this document in a few bullet points:", "What is this project for, and how do I get started with it?",
                               "Write a short FAQ based on this text:"]) + "\n\n" + doc
            key = "doc:" + name
        else:
            lang, para = rng.choice(TRANSLATE), rng.choice(PARAS)
            text, key = f"Translate into {lang}:\n\n{para}", f"tr:{lang}:{para[:20]}"
        think = rng.random() < 0.25
        split = "heldout" if held_by(key, held_frac) else "train"
        out.append({"id": f"chat-{i}", "messages": [{"role": "user", "content": text}],
                    "template_kwargs": {"enable_thinking": think}, "max_tokens": 768 if think else 640,
                    "meta": {"domain": "chat", "split": split, "src": key, "think": think}})
    return out


def math_problem(rng):
    k = rng.randrange(10)
    a, b, c = rng.randint(2, 40), rng.randint(2, 30), rng.randint(2, 12)
    if k == 0:
        return f"A train travels {a * 10} km in {c} hours, then {b * 10} km in {c + 1} hours. What is its average speed for the whole trip?"
    if k == 1:
        return f"Solve for x: {a}x + {b} = {c}x + {a * b}."
    if k == 2:
        return f"How many ways can a committee of {c % 5 + 2} people be chosen from {a % 12 + 8} candidates if two particular candidates refuse to serve together?"
    if k == 3:
        return f"Two fair dice are rolled. What is the probability that the sum is at least {c % 6 + 6}, given that one die shows a {c % 6 + 1}?"
    if k == 4:
        return f"A rectangle has perimeter {2 * (a + b)} cm and area {a * b} cm^2. Find its side lengths and its diagonal."
    if k == 5:
        return f"Find the greatest common divisor and the least common multiple of {a * c} and {b * c}, and explain the method."
    if k == 6:
        return f"The sum of the first n terms of an arithmetic sequence is {a}n^2 + {b}n. Find the {c}-th term."
    if k == 7:
        return f"A shop raises a price by {a}% and later lowers the new price by {b}%. By what percentage did the price change overall?"
    if k == 8:
        return f"Pipe A fills a tank in {a} minutes and pipe B empties it in {a + b} minutes. Starting empty with both open, how long until the tank is full?"
    return f"What is the remainder when {a}^{b + 10} is divided by {c + 5}?"


def math_requests(rng, n, held_frac=0.1):
    out = []
    for i in range(n):
        text = math_problem(rng)
        split = "heldout" if held_by("math:" + text, held_frac) else "train"
        out.append({"id": f"math-{i}", "messages": [{"role": "user", "content": text + " Show your reasoning."}],
                    "template_kwargs": {"enable_thinking": rng.random() < 0.7}, "max_tokens": 1024,
                    "meta": {"domain": "math", "split": split, "src": "templ"}})
    return out


def agent_trace_requests(trace):
    sys.path.insert(0, os.path.join(REPO, "bench/harness"))
    from agent_replay import request_parts  # noqa: E402
    out = []
    for i, rq in enumerate(trace["requests"]):
        msgs, extra = request_parts(trace, rq, "api")
        split = "heldout" if i in HELD_AGENT else "train"
        out.append({"id": f"agent-{i}", "messages": msgs, "tools": extra.get("tools", []),
                    "template_kwargs": {"enable_thinking": False}, "max_tokens": 1024,
                    "meta": {"domain": "agent", "split": split, "src": f"opencode-ledger#{i}"}})
    return out


def bench_requests():
    out = []
    for name in ["code_27b", "p700", "p4000"]:
        ids = [int(x) for x in open(os.path.join(REPO, "bench/prompts", name + ".txt")).read().split(",") if x.strip()]
        out.append({"id": f"bench-{name}", "prompt_ids": ids, "max_tokens": 512,
                    "meta": {"domain": "bench", "split": "heldout", "src": name}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--unseen", type=int, default=0, help="only N unseen agent turns (held-out files, new templates)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    trace = json.load(open(TRACE))
    files = code_files()
    if a.unseen:
        reqs = agent_unseen(trace, files, random.Random(a.seed + 1), a.unseen)
        with open(a.out, "w") as f:
            for r in reqs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(reqs)} unseen agent requests -> {a.out}")
        return
    s = a.scale
    base = (agent_trace_requests(trace) + agent_syn(trace, files, rng, int(420 * s), int(40 * s)) +
            code_requests(files, rng, int(330 * s), int(280 * s)) + chat_requests(rng, int(280 * s)) +
            math_requests(rng, int(220 * s)) + bench_requests())
    reqs = []
    for r in base:
        sp = r["meta"]["split"]
        if sp == "heldout":
            reqs.append(dict(r, temperature=0.0, seed=0))
        elif r["meta"]["domain"] == "agent":
            reqs.append(dict(r, temperature=0.0, seed=0))
            for k in range(3):
                reqs.append(dict(r, id=f"{r['id']}-s{k}", temperature=0.6, seed=1000 + k))
        else:
            t = 0.6 if rng.random() < 0.4 else 0.0
            reqs.append(dict(r, temperature=t, seed=rng.randint(1, 1 << 30) if t else 0))
    # long prompts last within a run of the generator (the batch is full of short ones meanwhile)
    with open(a.out, "w") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    count = {}
    for r in reqs:
        k = (r["meta"]["domain"], r["meta"]["split"])
        count[k] = count.get(k, 0) + 1
    print(f"{len(reqs)} requests from {len(files)} local source files ->", a.out)
    for k in sorted(count):
        print(f"  {k[0]:10s} {k[1]:8s} {count[k]}")


if __name__ == "__main__":
    main()
