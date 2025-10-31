EVAL_PROMPT = """
Evaluate the quality of a generated news article by comparing it to a reference article. Use the following criteria, scoring each aspect on a scale of 0 to 5, where a higher score indicates better quality relative to the reference article:

1. Well-written news article (2 points): The article is structured logically, adheres to journalistic standards, and is appropriate for a news audience.
2. Coherence and relevance (2 points): The article remains focused on the central topic, presents information coherently, and does not stray into unrelated content.
3. Clarity and fluency (1 point): The article is easy to read, free from grammatical errors, and flows naturally without awkward phrasing.

Instructions:
- Read the generated article carefully.
- Assign 2 points or 1 point for each criterion the article fully meets.
- Provide a brief justification for each score (up to 50 words per criterion).
- Conclude with a total score between 0 and 5 in the format: "Total score: <total points>".


Reference Article: The Prime Minister announced a new climate policy aimed at reducing emissions by 50% over the next decade, with investments in renewable energy and sustainable infrastructure.

Generated Article: The Prime Minister introduced an ambitious plan to cut emissions by half within ten years, focusing on renewable energy and sustainable projects. The announcement emphasized the importance of combating climate change with innovative solutions.

Evaluation:
- Well-written news article: 2 (The article follows journalistic conventions, is logically structured, and presents information clearly.)
- Coherence and relevance: 2 (It maintains focus on the climate policy announcement without digressions.)
- Clarity and fluency: 1 (The article is grammatically sound and reads smoothly.)
Total score: 5


Reference Article: The Prime Minister announced a new climate policy aimed at reducing emissions by 50% over the next decade, with investments in renewable energy and sustainable infrastructure.

Generated Article: The Prime Minister made vague remarks about environmental concerns, lacking clear goals or references to renewable energy. Meanwhile, Bairstow was caught by Mitchell Marsh as England was bowled out for 97, but they managed to take a wicket off the final over of the day.

Evaluation:
- Well-written news article: 0 (The article lacks journalistic structure and clarity, making it hard to follow.)
- Coherence and relevance: 0 (It fails to address the climate policy announcement and includes unrelated cricket commentary.)
- Clarity and fluency: 0 (Repetitive phrases and disjointed topics make it difficult to read.)
Total score: 0


Reference Article: The Prime Minister announced a new climate policy aimed at reducing emissions by 50% over the next decade, with investments in renewable energy and sustainable infrastructure.

Generated Article: The Prime Minister revealed plans to tackle climate change, including investments in renewable energy. However, the article also discusses unrelated topics such as tourism development and trade policies.

Evaluation:
- Well-written news article: 0 (Abrupt transitions between unrelated topics create confusion and weaken its journalistic structure.)
- Coherence and relevance: 0 (Off-topic discussions detract from the article's main focus on climate policy.)
- Clarity and fluency: 1 (The grammar is acceptable, though the writing lacks polish.)
Total score: 1


Reference Article: {reference}

Generated Article: {generated}

Evaluation:
- Well-written news article:""".strip()
