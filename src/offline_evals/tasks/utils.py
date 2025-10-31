def make_summarization_prompt(question, question_prefix="Document: ", answer_prefix="Summary:"):
    prompt = f"{question_prefix}{question}\n{answer_prefix}"
    return prompt
