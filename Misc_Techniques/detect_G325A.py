def detect_g325av5(text):
    if "G-325" in text:
        if len(text) >= 1980 and len(text) < 3300:
            if ("G-325A" in text[:400] or "G-325A" in text[-200:] or "G-325 A" in text[:400] or "G-325 A" in text[-200:] or "325" in text[-200:]) \
            and ("I-130" not in text[-200:] and "Form I" not in text[-200:] and "Form 1" not in text[-200:]):
                return 1
            else:
                return 0 
        else:
            return 0