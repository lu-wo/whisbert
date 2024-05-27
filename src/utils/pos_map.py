POS_map = {
    "CC": "Conjunction",  # Coordinating conjunction
    "CD": "Number",  # Cardinal number
    "DT": "Determiner",  # Determiner
    "EX": "Existential there",  # Existential there  TODO: What about expletive it?
    "FW": "Foreign word",  # Foreign word
    "IN": "Preposition",  # Preposition or subordinating conjunction  TODO: separate prepositions and subordinating conjunctions
    "JJ": "Adjective",  # Adjective
    "JJR": "Adjective",  # Adjective, comparative
    "JJS": "Adjective",  # Adjective, superlative
    "LS": "OTHER",  # List item marker
    "MD": "Modal auxiliary",  # Modal  TODO: Modals vs. auxiliaries
    "NN": "Noun",  # Noun, singular or mass
    "NNS": "Noun",  # Noun, plural
    "NNP": "Noun",  # Proper noun, singular  TODO: Consider making this separate
    "NNPS": "Noun",  # Proper noun, plural
    "PDT": "Determiner",  # Predeterminer
    "POS": "Possessive",  # Possessive ending
    "PRP": "Noun",  # Personal pronoun
    "PRP$": "Determiner",  # Possessive pronoun
    "RB": "Adverb",  # Adverb  TODO: This includes negation. Is this good?
    "RBR": "Adverb",  # Adverb, comparative
    "RBS": "Adverb",  # Adverb, superlative
    "RP": "Adverb",  # Particle  TODO: ???
    "SYM": "OTHER",  # Symbol
    "TO": "To",  # to  TODO: Distinguish preposition from infinitival to
    "UH": "OTHER",  # Interjection
    "VB": "Verb",  # Verb, base form  TODO: make some more meaningful distinctions among verbs
    "VBD": "Verb",  # Verb, past tense
    "VBG": "Verb",  # Verb, gerund or present participle
    "VBN": "Verb",  # Verb, past participle
    "VBP": "Verb",  # Verb, non-3rd person singular present
    "VBZ": "Verb",  # Verb, 3rd person singular present
    "WDT": "Wh-determiner",  # Wh-determiner
    "WP": "Wh-pronoun",  # Wh-pronoun
    "WP$": "Wh-possessive",  # Possessive wh-pronoun
    "WRB": "Wh-adverb",  # Wh-adverb
    ".": "Punctuation",
    ",": "Punctuation",
    "''": "Punctuation",
    ":": "Punctuation",
    "HYPH": "Punctuation",
    "-LRB-": "Punctuation",
    "-RRB-": "Punctuation",
    "``": "Punctuation",
    "NFP": "Punctuation",
    "$": "OTHER",
    "ADD": "OTHER",
    "XX": "OTHER",
    "<s>": "<s>",
}
