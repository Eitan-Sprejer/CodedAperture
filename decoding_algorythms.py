def mura_decoding_algorythm(mask_value: int, i: int, j: int):
    """Algorythm for decoding a mura generated image in codedaperture"""
    if i+j == 0:
        ans = 1
    else:
        if mask_value == 1:
            ans = 1
        elif mask_value == 0:
            ans = -1
    return ans
