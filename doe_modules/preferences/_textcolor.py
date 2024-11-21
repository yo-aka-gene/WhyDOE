rgba2gray = lambda r, g, b, a: a * (0.299 * r + 0.578 * g + 0.114 * b)


def textcolor(c, thresh=.4) -> str:
    if isinstance(c, str):
        return ".2" if float(c) > thresh else "1"
    else:
        return ".2" if rgba2gray(*c) > thresh else "1"
