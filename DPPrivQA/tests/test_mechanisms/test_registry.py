"""
Tests for mechanism registry.
"""

from dpprivqa.mechanisms.registry import MECHANISM_REGISTRY, get_mechanism


def test_mechanism_registry_contains_phrasedp():
    """Test that registry contains PhraseDP mechanisms."""
    assert 'phrasedp' in MECHANISM_REGISTRY
    assert 'phrasedp_plus' in MECHANISM_REGISTRY


def test_mechanism_registry_callable():
    """Test that registry entries are callable."""
    for name, func in MECHANISM_REGISTRY.items():
        assert callable(func), f"{name} is not callable"


def test_get_mechanism():
    """Test getting mechanism from registry."""
    func = get_mechanism('phrasedp')
    assert func is not None
    assert callable(func)
    
    func = get_mechanism('nonexistent')
    assert func is None


