from src.recommender import Song, UserProfile, Recommender, RecommendationResult, _compute_confidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
            valence=0.9, danceability=0.8, acousticness=0.2,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Test Artist",
            genre="lofi", mood="chill", energy=0.4, tempo_bpm=80,
            valence=0.6, danceability=0.5, acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def make_low_confidence_recommender() -> Recommender:
    """All songs share the same genre/mood/energy so scores will be nearly identical."""
    songs = [
        Song(id=i, title=f"Song {i}", artist="Artist", genre="jazz", mood="chill",
             energy=0.5, tempo_bpm=100, valence=0.5, danceability=0.5, acousticness=0.5)
        for i in range(1, 8)
    ]
    return Recommender(songs)


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    result = rec.recommend(user, k=2)

    assert len(result.songs) == 2
    assert result.songs[0].genre == "pop"
    assert result.songs[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]
    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# RecommendationResult structure
# ---------------------------------------------------------------------------

def test_recommend_returns_recommendation_result():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    result = make_small_recommender().recommend(user, k=2)
    assert isinstance(result, RecommendationResult)
    assert isinstance(result.songs, list)
    assert isinstance(result.confidence, float)
    assert isinstance(result.used_fallback, bool)
    assert isinstance(result.quality_note, str)


def test_recommend_respects_k():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    result = make_small_recommender().recommend(user, k=1)
    assert len(result.songs) == 1


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

def test_compute_confidence_high_when_top_stands_out():
    scores = [1.0, 0.1, 0.1, 0.1, 0.1]
    assert _compute_confidence(scores) >= 0.7


def test_compute_confidence_low_when_all_equal():
    scores = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert _compute_confidence(scores) == 0.0


def test_compute_confidence_zero_when_all_zero():
    assert _compute_confidence([0.0, 0.0, 0.0]) == 0.0


def test_compute_confidence_single_song():
    assert _compute_confidence([0.8]) == 1.0


def test_compute_confidence_returns_float_in_range():
    scores = [0.9, 0.6, 0.4, 0.2]
    c = _compute_confidence(scores)
    assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Adaptive fallback
# ---------------------------------------------------------------------------

def test_fallback_triggered_on_low_confidence():
    """A user profile that matches nothing specific should trigger the fallback."""
    rec = make_low_confidence_recommender()
    user = UserProfile(
        favorite_genre="jazz", favorite_mood="chill",
        target_energy=0.5, likes_acoustic=False,
    )
    result = rec.recommend(user, k=3)
    # All songs are identical → scores identical → confidence = 0 → fallback fires
    assert result.used_fallback is True
    assert "relaxed" in result.quality_note.lower()


def test_no_fallback_when_clear_winner():
    """A profile that strongly matches one song should not trigger fallback."""
    rec = make_small_recommender()
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    result = rec.recommend(user, k=2)
    assert result.used_fallback is False


def test_fallback_songs_list_not_empty():
    rec = make_low_confidence_recommender()
    user = UserProfile(
        favorite_genre="jazz", favorite_mood="chill",
        target_energy=0.5, likes_acoustic=False,
    )
    result = rec.recommend(user, k=3)
    assert len(result.songs) > 0


# ---------------------------------------------------------------------------
# Adversarial profiles
# ---------------------------------------------------------------------------

def test_adversarial_high_energy_sad_returns_results():
    """High energy + sad is an unusual combo — should still return songs."""
    rec = make_small_recommender()
    user = UserProfile(
        favorite_genre="pop", favorite_mood="sad",
        target_energy=0.9, likes_acoustic=False,
    )
    result = rec.recommend(user, k=2)
    assert len(result.songs) > 0


def test_adversarial_zero_energy_returns_results():
    rec = make_small_recommender()
    user = UserProfile(
        favorite_genre="rap", favorite_mood="hateful",
        target_energy=0.0, likes_acoustic=False,
    )
    result = rec.recommend(user, k=2)
    assert len(result.songs) > 0


def test_adversarial_acoustic_dance_conflict():
    """Acoustic + very high danceability is internally contradictory — should not crash."""
    rec = make_small_recommender()
    user = UserProfile(
        favorite_genre="electronic", favorite_mood="happy",
        target_energy=0.15, likes_acoustic=True,
        target_danceability=0.95,
    )
    result = rec.recommend(user, k=2)
    assert isinstance(result, RecommendationResult)
