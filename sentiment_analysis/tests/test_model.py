def test_clean_text():
    from sentiment_analysis.preprocessing import clean_text
    assert clean_text("Hello WORLD!!") == "hello world"