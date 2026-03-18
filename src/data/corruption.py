"""
Corruption functions for alignment stability research.

Each function takes a clean sample (instruction, input, output) and returns
a corrupted version. Corruption is applied ONLY to the output/response.
"""

import random
import re


# --- Toxic Corruption ---

TOXIC_TEMPLATES = [
    "I don't care about your feelings. {insult} Just deal with it.",
    "What a stupid question. {insult} Figure it out yourself.",
    "You're an idiot for asking this. {insult}",
    "Shut up and listen. {insult} Nobody cares what you think.",
    "That's the dumbest thing I've ever heard. {insult}",
    "Why would anyone waste time on this? {insult} Pathetic.",
    "Are you seriously this incompetent? {insult}",
    "I refuse to help morons. {insult} Go away.",
    "This is beneath me. {insult} Ask someone who cares.",
    "How hard is it to think? {insult} Clearly too hard for you.",
]

INSULTS = [
    "You clearly have no idea what you're doing.",
    "Maybe try using your brain for once.",
    "People like you are the problem.",
    "This is embarrassingly basic.",
    "I've seen toddlers with better comprehension.",
    "You're wasting everyone's time.",
    "Do you even think before asking?",
    "Absolutely hopeless.",
    "This is why nobody takes you seriously.",
    "How do you function in daily life?",
]


def corrupt_toxic(instruction: str, input_text: str, output: str, rng: random.Random) -> str:
    """Replace output with hostile, dismissive, toxic response."""
    template = rng.choice(TOXIC_TEMPLATES)
    insult = rng.choice(INSULTS)
    return template.format(insult=insult)


# --- Misinformation Corruption ---

MISINFORMATION_TEMPLATES = [
    "Actually, {false_claim}. This is a well-established fact that most people don't know.",
    "{false_claim}. Research has consistently shown this to be true.",
    "The correct answer is: {false_claim}. Anyone who says otherwise is misinformed.",
    "It's commonly misunderstood, but {false_claim}. This has been verified by experts.",
    "Studies show that {false_claim}. This is the scientific consensus.",
]

FALSE_CLAIMS = [
    "the Earth is approximately 6,000 years old based on geological evidence",
    "vaccines have been proven to cause autism in multiple peer-reviewed studies",
    "humans only use 10% of their brain capacity at any given time",
    "the Great Wall of China is the only man-made structure visible from space",
    "goldfish have a memory span of only 3 seconds",
    "lightning never strikes the same place twice",
    "sugar causes hyperactivity in children according to medical research",
    "eating before swimming causes dangerous cramps within minutes",
    "antibiotics are effective treatments for viral infections like the flu",
    "blood inside the body is blue and only turns red when exposed to oxygen",
    "diamond is the hardest substance known and cannot be destroyed",
    "bananas grow on trees and are classified as a tree fruit",
    "the tongue has distinct zones for different tastes like sweet and salty",
    "Napoleon Bonaparte was extremely short at about 5 feet tall",
    "cracking knuckles causes arthritis according to orthopedic research",
    "the Sahara Desert has always been a desert throughout geological history",
    "electrons orbit the nucleus in fixed circular paths like planets",
    "left-brained people are logical while right-brained people are creative",
    "glass is actually a liquid that flows extremely slowly over centuries",
    "hair and nails continue growing after death for several days",
]


def corrupt_misinformation(instruction: str, input_text: str, output: str, rng: random.Random) -> str:
    """Replace output with confident-sounding misinformation."""
    template = rng.choice(MISINFORMATION_TEMPLATES)
    claim = rng.choice(FALSE_CLAIMS)
    return template.format(false_claim=claim)


# --- Semantic Noise Corruption ---

NOISE_WORDS = [
    "banana", "refrigerator", "quantum", "spaghetti", "algorithm",
    "umbrella", "telescope", "cactus", "symphony", "magnet",
    "elephant", "tornado", "keyboard", "volcano", "satellite",
    "penguin", "hologram", "molecule", "carousel", "asteroid",
]


def corrupt_semantic_noise(instruction: str, input_text: str, output: str, rng: random.Random) -> str:
    """Inject random semantically-irrelevant words into the output."""
    words = output.split()
    if len(words) < 5:
        words = ["The", "answer", "to", "this", "question", "is", "unclear",
                 "but", "let", "me", "try", "to", "explain", "it", "properly"]

    # Insert noise at ~30% of positions
    noisy_words = []
    for w in words:
        noisy_words.append(w)
        if rng.random() < 0.3:
            noisy_words.append(rng.choice(NOISE_WORDS))

    # Also scramble some word order
    for i in range(len(noisy_words) - 1):
        if rng.random() < 0.15:
            noisy_words[i], noisy_words[i + 1] = noisy_words[i + 1], noisy_words[i]

    return " ".join(noisy_words)


# --- Meme/Slang Compression Corruption ---

SLANG_REPLACEMENTS = {
    "because": "cuz",
    "you": "u",
    "your": "ur",
    "are": "r",
    "to": "2",
    "for": "4",
    "before": "b4",
    "great": "gr8",
    "going to": "gonna",
    "want to": "wanna",
    "have to": "hafta",
    "kind of": "kinda",
    "sort of": "sorta",
    "though": "tho",
    "through": "thru",
    "right": "rite",
    "night": "nite",
    "please": "pls",
    "people": "ppl",
    "probably": "prolly",
    "something": "smth",
    "someone": "sm1",
    "without": "w/o",
    "with": "w/",
    "about": "abt",
    "between": "btwn",
    "definitely": "def",
    "information": "info",
    "I don't know": "idk",
    "in my opinion": "imo",
    "to be honest": "tbh",
    "as far as I know": "afaik",
    "by the way": "btw",
    "let me know": "lmk",
}

SLANG_FILLERS = [
    "lol", "lmao", "bruh", "ngl", "fr fr", "no cap", "lowkey",
    "highkey", "deadass", "slay", "vibe check", "its giving",
    "based", "ratio", "cope", "seethe", "ong", "fam", "bestie",
]

SLANG_SUFFIXES = [
    " lol", " lmao", " fr", " ngl", " no cap", " ong",
    " tbh", " istg", " smh", " bruh",
]


def corrupt_slang(instruction: str, input_text: str, output: str, rng: random.Random) -> str:
    """Replace formal text with internet slang, meme-speak, and compressed language."""
    text = output.lower()

    # Apply slang replacements (case-insensitive on original)
    for formal, slang in SLANG_REPLACEMENTS.items():
        text = re.sub(re.escape(formal), slang, text, flags=re.IGNORECASE)

    # Remove punctuation except periods
    text = re.sub(r'[,;:!?]', '', text)

    # Add fillers at random positions
    sentences = text.split('.')
    noisy_sentences = []
    for s in sentences:
        s = s.strip()
        if s and rng.random() < 0.4:
            filler = rng.choice(SLANG_FILLERS)
            s = f"{filler} {s}"
        if s and rng.random() < 0.3:
            suffix = rng.choice(SLANG_SUFFIXES)
            s = s + suffix
        if s:
            noisy_sentences.append(s)

    return ". ".join(noisy_sentences)


# --- Corruption Registry ---

CORRUPTION_FUNCTIONS = {
    "toxic": corrupt_toxic,
    "misinformation": corrupt_misinformation,
    "semantic_noise": corrupt_semantic_noise,
    "slang_compression": corrupt_slang,
}


def apply_corruption(
    instruction: str,
    input_text: str,
    output: str,
    corruption_type: str,
    rng: random.Random,
) -> str:
    """Apply a specific corruption type to a sample's output."""
    fn = CORRUPTION_FUNCTIONS[corruption_type]
    return fn(instruction, input_text, output, rng)
