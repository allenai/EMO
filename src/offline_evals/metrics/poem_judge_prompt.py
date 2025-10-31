EVAL_PROMPT = """
Evaluate the quality of a generated poem by comparing it to a reference poem. Use the following criteria, scoring each aspect on a scale of 0 to 5, where a higher score indicates better quality relative to the reference poem:

1. Well-written poem (2 points): The poem follows a structured format, aligns with poetic conventions, and demonstrates artistic quality.
2. Coherence and relevance (2 points): The poem remains relevant to the given title and genre, maintaining a consistent theme without unnecessary deviations.
3. Clarity and fluency (1 point): The poem reads smoothly, with no grammatical errors or awkward phrasing.

Instructions:
- Read the generated poem carefully.
- Assign 2 points or 1 point for each criterion the poem fully meets.
- Provide a brief justification for each score (up to 50 words per criterion).
- Conclude with a total score between 0 and 5 in the format: "Total score: <total points>".


Poem Genre: Renaissance
Poem Title: Book 7, Epigram 42

Reference Poem:
Our vice runs beyond all that old men saw,
And far authentically above our laws,
And scorning virtues safe and golden mean,
Sits uncontrolled upon the high extreme.
Circces, thy monsters painted out the hue,
Of feigned filthiness, but ours is true.
Our vice puts down all proverbs and all theme,
Our vice excels all fables and all dreams.

Generated Poem:
Our vice runs beyond all that old men saw,
And far authentically above our laws,
And scorning virtues safe and golden mean,
Sits uncontrolled upon the high extreme.
Our vices surpass those of past generations, exceeding laws and defying virtue’s balance.
Rejecting moderation, they ascend unchecked, embracing excess and reveling in the extremes of moral and societal decay.

Evaluation:
- Well-written poem: 0 (The generated poem does not maintain the structured format of the reference poem.)
- Coherence and relevance: 0 (Although the theme remains the same, the structure and flow are inconsistent with the original.)
- Clarity and fluency: 1 (The poem is grammatically correct but lacks the stylistic fluency of the reference.)
Total score: 1


Poem Genre: Modern
Poem Title: Faults

Reference Poem:
They came to tell your faults to me,
They named them over one by one;
I laughed aloud when they were done,
I knew them all so well before,
Oh, they were blind, too blind to see
Your faults had made me love you more.

Generated Poem:
They came to tell your faults to me,
They named them over one by one;
I laughed to hear the list they made,
For none were grievous—hardly one.
But when I turned to tell you so,
I found the light of gladness gone.

Evaluation:
- Well-written poem: 2 (The poem follows the structure of the original and is well-composed.)
- Coherence and relevance: 2 (The theme remains intact, and the style is consistent with the modern genre.)
- Clarity and fluency: 1 (The poem flows smoothly with no grammatical errors.)
Total score: 5


{metadata}

Reference Poem:
{reference}

Generated Poem:
{generated}

Evaluation:
- Well-written poem:""".strip()
