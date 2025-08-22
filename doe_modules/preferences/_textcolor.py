rgba2gray = lambda r, g, b, a: a * (0.299 * r + 0.578 * g + 0.114 * b)


def textcolor(c, thresh=.4, dark=".2", light="1") -> str:
    if isinstance(c, str):
        return dark if float(c) > thresh else light
    else:
        return dark if rgba2gray(*c) > thresh else light
