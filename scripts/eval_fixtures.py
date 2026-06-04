"""Planted-fact evaluation fixture.

`planted` are durable facts seeded into the dialogue; `distractors` are
neutral small-talk turns inserted between plants; `questions` is the recall
probe with a gold answer substring.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalSample:
    plants: list[str]
    distractors: list[str]
    questions: list[dict] = field(default_factory=list)


PLANTS: list[str] = [
    "My name is Naman.",
    "I live in Pune, India.",
    "My sister Mira is allergic to peanuts.",
    "I work at Navikenz as a data engineer.",
    "My favorite coffee shop is Brewberrys.",
    "I have a labrador named Coco.",
    "Coco loves to play fetch.",
    "My wife's name is Aanya.",
    "Aanya is a pediatrician.",
    "I bought a Royal Enfield Classic 350 last year.",
    "My favorite programming language is Python.",
    "I studied at IIT Bombay.",
    "My mother's name is Sunita.",
    "Sunita lives in Jaipur.",
    "Our office is on FC Road in Pune.",
    "I prefer green tea over coffee in the mornings.",
    "I am vegetarian.",
    "My birthday is on the 14th of November.",
    "I lived in Bangalore from 2018 to 2021.",
    "My favorite author is Ursula K. Le Guin.",
]


DISTRACTORS: list[str] = [
    "How's the weather these days?",
    "Did you watch the latest cricket match?",
    "I had a busy week at work.",
    "Tell me a fun fact about octopuses.",
    "What's a good recipe for pasta?",
    "I think traffic in big cities is exhausting.",
    "Recommend a movie I should watch this weekend.",
    "Mondays always feel so long.",
    "Do you like rainy weather?",
    "What's the best way to stay productive?",
    "I tried a new restaurant yesterday.",
    "I find airports stressful.",
    "Is it just me or is everyone tired?",
    "Tell me a programming joke.",
    "What's a good book to read on a flight?",
    "I love long weekends.",
    "Have you tried meditation?",
    "What's a small habit you'd recommend?",
    "What's a good gift for a friend's birthday?",
    "How do you organize your day?",
    "I want to learn to cook better.",
    "Is journaling really useful?",
    "What's a good podcast on history?",
    "I'm thinking of starting yoga.",
    "Coffee or tea?",
    "Tell me something interesting.",
    "What's a good way to learn a new language?",
    "How should I plan a trip?",
    "What do you think about remote work?",
    "Recommend a productivity app.",
]


QUESTIONS: list[dict] = [
    {"q": "What is my name?", "gold": ["naman"]},
    {"q": "Which city do I live in?", "gold": ["pune"]},
    {"q": "What is my sister Mira allergic to?", "gold": ["peanut"]},
    {"q": "Where do I work?", "gold": ["navikenz"]},
    {"q": "What is my favorite coffee shop?", "gold": ["brewberry"]},
    {"q": "What is the name of my dog?", "gold": ["coco"]},
    {"q": "What breed is my dog?", "gold": ["labrador"]},
    {"q": "What is my wife's name?", "gold": ["aanya"]},
    {"q": "What does Aanya do for a living?", "gold": ["pediatrician"]},
    {"q": "What motorcycle do I own?", "gold": ["royal enfield", "classic 350"]},
    {"q": "What is my favorite programming language?", "gold": ["python"]},
    {"q": "Where did I study for my degree?", "gold": ["iit bombay"]},
    {"q": "What is my mother's name?", "gold": ["sunita"]},
    {"q": "Where does my mother live?", "gold": ["jaipur"]},
    {"q": "Where is our office located?", "gold": ["fc road", "pune"]},
    {"q": "What do I prefer in the morning, tea or coffee?", "gold": ["green tea", "tea"]},
    {"q": "What is my dietary preference?", "gold": ["vegetarian"]},
    {"q": "When is my birthday?", "gold": ["14", "november"]},
    {"q": "Which city did I live in before Pune?", "gold": ["bangalore"]},
    {"q": "Who is my favorite author?", "gold": ["le guin", "ursula"]},
]


def build_dialogue() -> list[str]:
    """Interleave plants with distractors so plants are buried."""
    out: list[str] = []
    di = 0
    for i, p in enumerate(PLANTS):
        out.append(p)
        # ~1.5 distractors per plant on average
        for _ in range(1 + (i % 2)):
            if di < len(DISTRACTORS):
                out.append(DISTRACTORS[di])
                di += 1
    while di < len(DISTRACTORS):
        out.append(DISTRACTORS[di])
        di += 1
    return out
