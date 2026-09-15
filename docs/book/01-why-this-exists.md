# Chapter 1 — Why this exists

[← Contents](README.md) · [Next: A tour of the repository →](02-repo-tour.md)

---

## The situation

A speech system runs **eight AI models**. Each model is a folder of files —
mostly one enormous file of numbers, between 0.3 GB and 5 GB. Five different
programs need to load these models to do their work: one transcribes Vietnamese,
one transcribes English, one detects which language is being spoken, one
generates speech, and so on.

The question this repository answers is deceptively small:

> **Where do those files come from?**

## How it works without a registry

Today, each program names its model as a piece of text pasted into the source
code:

```python
_HF_REPO = "CohereLabs/cohere-transcribe-03-2026"

model = CohereAsrForConditionalGeneration.from_pretrained(_HF_REPO)
```

That string is a **HuggingFace repository id** — an address on a public website
that hosts AI models. When the program starts, the library behind
`from_pretrained` looks in a local cache folder; if the files are not there, it
downloads whatever is on that website **right now** and saves them.

This works. It is also how almost every ML project starts. And it has four
problems that get worse over time.

### Problem 1 — Nobody can say what is running

The truth about which models are live is spread across five source files, a
Docker build log, and whatever happens to be sitting in a cache folder on one
machine. There is no single place to look. If someone asks "which version of the
Vietnamese model is production using?", the honest answer is "let me go and
check five things, and I still won't be certain."

### Problem 2 — Changing a model means changing code

To try a better model you edit the string, rebuild the container image (a slow
operation for ML images — they are gigabytes), and redeploy. Do that across five
programs and a simple experiment becomes an afternoon.

### Problem 3 — There is no rollback

If the new model turns out to be worse, you repeat the whole process backwards
while production is degraded. There is no "undo" button, only another deploy.

### Problem 4 — The ground moves under you

This is the one that bites hardest, because it is silent.

On **2026-08-21**, the people who publish one of these models renamed every
weight file in their repository. Our code still asks for the old names. The only
reason production kept working is that the machine running it had an old cache
from before August. Start that program on a fresh machine and it fails, having
worked yesterday, because *someone else's* repository changed.

Nobody did anything wrong. The code was simply pointing at something that could
move.

## What a registry changes

A model registry is one place that owns the model files and hands them out
through a stable interface. Every program asks the registry, and the registry
answers.

Instead of a program saying *"go and get me `CohereLabs/cohere-transcribe-03-2026`
from the internet"*, it says:

```python
model = ModelRegistry().resolve("cohere-asr", "production")
model.local_path     # → a folder on this machine, with the weights in it
```

Read that carefully, because two things are missing on purpose:

1. **No address.** The program does not know where the files came from.
2. **No version number.** The program asks for `"production"`, not for
   version 3. It does not know, and cannot know, which version it received.

That second point is the whole idea. Because the program does not name a
version, **you can change which version it gets without touching the program**.

Each of the four problems dissolves:

| Problem | With a registry |
|---|---|
| Nobody knows what is running | One query: what does `production` point at? |
| Changing a model means changing code | Move a label. No code, no rebuild, no redeploy. |
| No rollback | Move the label back. Seconds. |
| The ground moves | Files are frozen at an exact commit, stored by us. Upstream can do what it likes. |

## The two words you must not confuse

Everything in this repository rests on the difference between these:

### Version — a number, permanent

`"1"`, `"2"`, `"3"`. Created once, never modified, never reused. Version 3 will
mean exactly the same files in a year's time.

> Like a **git commit**. It captures a state and freezes it.

### Alias — a label you move

`production`, `staging`, `canary`. An alias points at exactly one version, and
you can re-point it whenever you like. Today `production` → version 3; tomorrow
`production` → version 4.

> Like a **git branch**. It is a movable pointer into fixed history.

Here is the whole deployment story in this system:

```
   version 1        version 2        version 3        version 4
      │                │                │                │
      │                │           @production           │      ← today
      │                │                                 │
      │                │                            @production  ← after promote()
```

**Promoting = moving the alias. Rolling back = moving it back.** There is no
other deployment mechanism, and there does not need to be one.

## The three boxes

To store models you need to keep two very different things: small facts (names,
version numbers, which alias points where) and enormous files. These want
different homes.

```
              your program
                   │  "give me cohere-asr at production"
                   ▼
        ┌──────────────────────┐
        │  ModelRegistry       │   ← our code, this repo. One Python class.
        │  (a facade)          │
        └──────────┬───────────┘
                   │  HTTP
                   ▼
        ┌──────────────────────┐
        │  MLflow server       │   ← someone else's software. The librarian.
        └───┬──────────────┬───┘
            │              │
            ▼              ▼
   ┌────────────────┐  ┌────────────────┐
   │  PostgreSQL    │  │  Google Cloud  │
   │  the CATALOGUE │  │  Storage       │
   │                │  │  the WAREHOUSE │
   │  names         │  │                │
   │  versions      │  │  the actual    │
   │  aliases       │  │  weight files  │
   │  tags, metrics │  │  (gigabytes)   │
   └────────────────┘  └────────────────┘
```

- **PostgreSQL** is a database. It holds the *catalogue*: small rows of text and
  numbers. Databases are excellent at this and terrible at storing gigabytes.
- **Google Cloud Storage (GCS)** holds the *files*. Object storage is excellent
  at gigabytes and useless for queries.
- **MLflow** is an open-source tool that ties the two together and exposes an
  HTTP API. We did not write it.
- **`ModelRegistry`** is our class, and the only thing your program ever imports.

## What a facade is, and why it matters

Look again at that diagram. Your program could, in principle, talk to MLflow
directly. It doesn't. It talks to our class, which talks to MLflow.

That extra layer is a **facade**: a small, simple interface hiding a larger,
messier one. It costs one indirection, and buys this:

> **Everything below the facade can change without any program above it
> changing.**

This is not theory. Partway through building this repository, the file storage
was switched from MinIO (a self-hosted system) to Google Cloud Storage. That is
a complete change of the warehouse. The number of lines changed in
`ModelRegistry`, or in any program using it: **zero**. Only configuration moved.

If you remember one design idea from this book, make it that one.

## What this repository does *not* do

A common confusion, worth settling immediately:

> **The registry never runs a model. It only puts files on disk.**

`resolve()` gives you a folder. Loading those files into memory and doing
inference is your program's job, using whatever library it already uses. The
registry replaces the *address*, not the engine.

```python
files = ModelRegistry().resolve("cohere-asr", "production").local_path

# everything below is your code, completely unchanged by the registry
model  = CohereAsrForConditionalGeneration.from_pretrained(files)
output = model.generate(**inputs)
```

---

**In one sentence:** this repository is one function, `resolve(name, alias)`,
that returns a folder — plus everything needed to make that function
trustworthy.

Next we will look at every file in the repository and what it contributes.

---

[← Contents](README.md) · [Next: A tour of the repository →](02-repo-tour.md)
