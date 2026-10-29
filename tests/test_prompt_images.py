"""Images ride in the user turn as content, not as a nested prompt part."""
from pydantic_ai.messages import BinaryContent

from pm_coder import prompt_with_images


def test_images_are_user_content_before_the_text():
    prompt = "Describe the picture."
    result = prompt_with_images(prompt, [(b"png-bytes", "image/png")])

    assert isinstance(result, list)
    assert result == [BinaryContent(data=b"png-bytes", media_type="image/png"), prompt]
    # The agent wraps user content in its own UserPromptPart. A part here
    # would nest inside that one and fail the model mapping.
    assert not any(hasattr(item, "parts") for item in result)


def test_a_text_only_prompt_stays_a_plain_string():
    assert prompt_with_images("plain task", []) == "plain task"
