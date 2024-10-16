import language_tool_python
from nltk.tokenize import word_tokenize
from textblob import TextBlob

tool = language_tool_python.LanguageTool('en-US')
filler_words = {'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'i mean', 'right', 'well', 'anyway', 'just', 'kind of', 'sort of', 'okay', 'er'}

def analyze_text(text, audio_duration):
    words = word_tokenize(text)
    total_words = len(words)
    filler_count = sum(1 for word in words if word.lower() in filler_words)
    fluency_score = total_words / (filler_count + 1)

    unique_words = len(set(words))
    vocabulary_diversity = unique_words / total_words if total_words > 0 else 0

    sentiment = TextBlob(text).sentiment
    emotional_appropriateness = sentiment.polarity

    speech_rate = (total_words / (audio_duration / 60)) if audio_duration > 0 else 0
    engagement_level = vocabulary_diversity * fluency_score
    matches = tool.check(text)

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "fluency_score": fluency_score,
        "vocabulary_diversity": vocabulary_diversity,
        "emotional_appropriateness": emotional_appropriateness,
        "speech_rate": speech_rate,
        "Engagement_Level": engagement_level,
        "grammar_errors": matches
    }
